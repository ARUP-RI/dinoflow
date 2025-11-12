#!/usr/bin/env python
"""
Single-tube DinoFlow model inference script.
Loads a PyTorch Lightning checkpoint trained on a single tube and runs inference on test data.
"""

import logging
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, roc_auc_score
)

# Add parent directory to path to import dinoflow modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dinoflow.models import TubeEncoder, munge_state_dict, IlseBagModel
from dinoflow.data import TubeData

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ClassificationHead(nn.Module):
    """Simple classification head for DinoFlow models."""
    def __init__(self, num_features, num_classes, output_scale_factor=1.0):
        super().__init__()
        self.output_scale_factor = output_scale_factor
        self.layers = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.GELU(),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x):
        layer_dtype = self.layers[0].weight.dtype
        if x.dtype != layer_dtype:
            x = x.to(layer_dtype)
        return self.layers(x) * self.output_scale_factor


class CombinedModel(nn.Module):
    """Combined backbone + classifier model."""
    def __init__(self, backbone, classifier, freeze_backbone=False):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        if freeze_backbone:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.classifier(self.backbone(x.float()))


def load_model_from_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """
    Load a single-tube DinoFlow model from a PyTorch Lightning checkpoint.
    
    Args:
        checkpoint_path: Path to the .ckpt file
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        Loaded model in eval mode
    """
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    hparams = ckpt['hyper_parameters']
    model_conf = hparams.get('model_conf', {})
    
    logger.info(f"Model config: {model_conf}")
    logger.info(f"Backbone class: {hparams.get('backbone_class')}")
    
    # Reconstruct model based on backbone class
    if hparams.get('backbone_class') == 'CombinedModel':
        backbone = TubeEncoder(
            num_features=model_conf['num_features'],
            model_embed_dim=model_conf['model_embed_dim'],
            layers=model_conf['layers'],
            heads=model_conf['heads'],
            d_ff=model_conf['d_ff'],
            layertype=model_conf['layertype'],
        )
        classifier = ClassificationHead(
            model_conf['model_embed_dim'], 
            num_classes=1, 
            output_scale_factor=model_conf.get('output_scale_factor', 1.0)
        )
        model = CombinedModel(backbone, classifier, freeze_backbone=True)
    
    elif hparams.get('backbone_class') == 'IlseBagModel':
        model = IlseBagModel(
            num_features=model_conf['num_features'],
            model_embed_dim=model_conf['model_embed_dim'],
            output_classes=model_conf['output_classes'],
            proto_dim=model_conf['proto_dim'],
            bag_classes=model_conf['bag_classes'],
        )
    
    elif hparams.get('backbone_class') == 'ClassificationHead':
        model = ClassificationHead(
            num_features=model_conf['num_features'],
            num_classes=model_conf['num_classes'],
            output_scale_factor=model_conf['output_scale_factor'],
        )
    
    else:
        raise ValueError(f"Unknown backbone class: {hparams.get('backbone_class')}")
    
    # Load state dict
    model.load_state_dict(munge_state_dict(ckpt['state_dict']), strict=True)
    model.eval()
    model.to(device)
    
    logger.info(f"Model loaded successfully on {device}")
    return model


def run_inference(
    checkpoint_path: str,
    test_csv: str,
    tube_type: str = 'b',
    labelkey: str = None,
    dataroot: str = '.',
    events: int = 8192,
    batch_size: int = 16,
    output_csv: str = None,
    device: str = None,
    task_type: str = 'auto'
):
    """
    Run inference on test data using a trained single-tube model.
    
    Args:
        checkpoint_path: Path to PyTorch Lightning checkpoint (.ckpt)
        test_csv: Path to test CSV file
        tube_type: Tube type ('b', 't', or 'm')
        labelkey: Column name for labels in CSV (None for inference-only mode)
        dataroot: Root directory for data files
        events: Number of events to sample per tube
        batch_size: Batch size for inference
        output_csv: Path to save predictions (optional)
        device: Device to use ('cuda' or 'cpu', auto-detect if None)
        task_type: Task type ('binary', 'regression', or 'auto' to detect from checkpoint)
    
    Returns:
        DataFrame with predictions and metrics dictionary (metrics will be None if no labels)
    """
    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Using device: {device}")
    
    # Load checkpoint to detect task type
    if task_type == 'auto':
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model_class = ckpt['hyper_parameters'].get('model_class', 'BinaryClassificationModel')
        if 'Regression' in model_class:
            task_type = 'regression'
            logger.info("Detected task type: REGRESSION")
        else:
            task_type = 'binary'
            logger.info("Detected task type: BINARY CLASSIFICATION")
    
    # Load model
    model = load_model_from_checkpoint(checkpoint_path, device=device)
    
    # Load test data
    logger.info(f"Loading test data from: {test_csv}")
    
    # Read CSV to allow inference-only mode (no labels)
    df = pd.read_csv(test_csv)
    has_labels = labelkey is not None and (labelkey in df.columns)
    if not has_labels:
        logger.info("Running in INFERENCE-ONLY mode (no ground truth labels found)")
        # Create a dummy label column so TubeData can operate
        dummy_label_key = 'label' if labelkey is None else labelkey
        if dummy_label_key not in df.columns:
            df[dummy_label_key] = 0.0
        labelkey = dummy_label_key
    
    testdata = TubeData(
        df, 
        data_root=dataroot, 
        labelkey=labelkey,
        tubes_to_return=[tube_type], 
        events_to_return=int(events)
    )
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=4)
    logger.info(f"Loaded {len(testloader.dataset)} test samples")
    
    # Run inference
    all_raw_outputs = []
    all_final_predictions = []
    all_labels = []
    all_accessions = []
    
    logger.info("Running inference...")
    with torch.inference_mode():
        for batch, rowdict in tqdm(testloader, desc="Processing batches"):
            labels = rowdict['label']
            accessions = rowdict['ACCESSION']
            
            # Get raw model outputs
            raw_output = model(batch.to(device)).cpu()
            
            # Process outputs based on task type
            if task_type == 'regression':
                # For regression: apply sigmoid and scale to 0-100 (matching training)
                final_pred = torch.sigmoid(raw_output).squeeze() * 100.0
            else:
                # For binary classification: apply sigmoid to get probabilities
                final_pred = torch.sigmoid(raw_output).squeeze()
            
            # Store results - handle both single and batch outputs
            if raw_output.dim() == 0 or (raw_output.dim() == 1 and raw_output.size(0) == 1):
                # Single sample
                all_raw_outputs.append(raw_output.item())
                all_final_predictions.append(final_pred.item())
            else:
                # Batch of samples
                all_raw_outputs.extend(raw_output.squeeze().numpy().tolist())
                all_final_predictions.extend(final_pred.numpy().tolist() if final_pred.dim() > 0 else [final_pred.item()])
            
            all_labels.extend(labels.numpy().tolist())
            all_accessions.extend(accessions)
    
    # Create results DataFrame with detailed scores
    if task_type == 'regression':
        if has_labels:
            results_df = pd.DataFrame({
                'ACCESSION': all_accessions,
                'tube_type': tube_type,
                'true_value': all_labels,
                'raw_logit': all_raw_outputs,
                'predicted_value': all_final_predictions
            })
        else:
            results_df = pd.DataFrame({
                'ACCESSION': all_accessions,
                'tube_type': tube_type,
                'raw_logit': all_raw_outputs,
                'predicted_value': all_final_predictions
            })
    else:
        results_df = pd.DataFrame({
            'ACCESSION': all_accessions,
            'tube_type': tube_type,
            'true_label': all_labels,
            'logit_score': all_raw_outputs,
            'probability_score': all_final_predictions,
            'predicted_label_0.5': [1 if p > 0.5 else 0 for p in all_final_predictions]
        })
    
    # Calculate metrics
    logger.info("\n" + "="*60)
    logger.info("EVALUATION METRICS")
    logger.info("="*60)
    
    # If no labels are available, skip metric computation
    if not has_labels:
        logger.info("No labels provided; skipping metric computation. Returning predictions only.")
        metrics = None
    else:
        y_true = np.array(all_labels)
        y_pred = np.array(all_final_predictions)
        
        if task_type == 'regression':
            # Regression metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            logger.info(f"Mean Squared Error (MSE): {mse:.4f}")
            logger.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            logger.info(f"Mean Absolute Error (MAE): {mae:.4f}")
            logger.info(f"R² Score: {r2:.4f}")
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
        else:
            # Classification metrics
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            # Basic metrics
            accuracy = (y_true == y_pred_binary).mean()
            logger.info(f"Accuracy: {accuracy:.4f}")
            
            # ROC curve and AUC
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            logger.info(f"ROC AUC: {roc_auc:.4f}")
            
            # Precision-Recall curve
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred)
            avg_precision = average_precision_score(y_true, y_pred)
            logger.info(f"Average Precision (AUPRC): {avg_precision:.4f}")
            
            # Sensitivity at specific FPR thresholds
            fpr_thresholds = [0.01, 0.02, 0.05]
            logger.info("\nSensitivity at FPR thresholds:")
            for fpr_threshold in fpr_thresholds:
                idx = fpr <= fpr_threshold
                if np.any(idx):
                    sensitivity = tpr[idx].max()
                    threshold_val = thresholds[idx][tpr[idx].argmax()]
                    logger.info(f"  FPR ≤ {fpr_threshold}: Sensitivity = {sensitivity:.4f}, Threshold = {threshold_val:.4f}")
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
            logger.info(f"\nConfusion Matrix (threshold=0.5):")
            logger.info(f"  True Negatives:  {tn}")
            logger.info(f"  False Positives: {fp}")
            logger.info(f"  False Negatives: {fn}")
            logger.info(f"  True Positives:  {tp}")
            
            # Classification report
            logger.info("\nClassification Report:")
            logger.info("\n" + classification_report(y_true, y_pred_binary, target_names=['Negative', 'Positive']))
            
            # Compile metrics dictionary
            metrics = {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'avg_precision': avg_precision,
                'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
            }
    
    # Save predictions if requested
    if output_csv:
        results_df.to_csv(output_csv, index=False)
        logger.info(f"\nPredictions saved to: {output_csv}")
    
    return results_df, metrics


def main():
    parser = argparse.ArgumentParser(
        description='Run inference on test data using a trained single-tube DinoFlow model'
    )
    parser.add_argument('checkpoint', type=str, help='Path to PyTorch Lightning checkpoint (.ckpt)')
    parser.add_argument('test_csv', type=str, help='Path to test CSV file')
    parser.add_argument('--tube-type', type=str, default='b', choices=['b', 't', 'm'],
                        help='Tube type to use (default: b)')
    parser.add_argument('--labelkey', type=str, default='label',
                        help='Column name for labels in CSV (default: label)')
    parser.add_argument('--dataroot', type=str, default='.',
                        help='Root directory for data files (default: .)')
    parser.add_argument('--events', type=int, default=8192,
                        help='Number of events to sample per tube (default: 8192)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for inference (default: 16)')
    parser.add_argument('--output-csv', type=str, default=None,
                        help='Path to save predictions CSV (optional)')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'],
                        help='Device to use (default: auto-detect)')
    parser.add_argument('--task-type', type=str, default='auto', choices=['auto', 'binary', 'regression'],
                        help='Task type: binary classification or regression (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Run inference
    results_df, metrics = run_inference(
        checkpoint_path=args.checkpoint,
        test_csv=args.test_csv,
        tube_type=args.tube_type,
        labelkey=args.labelkey,
        dataroot=args.dataroot,
        events=args.events,
        batch_size=args.batch_size,
        output_csv=args.output_csv,
        device=args.device,
        task_type=args.task_type
    )
    
    logger.info("\nInference complete!")


if __name__ == '__main__':
    main()

