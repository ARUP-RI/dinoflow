import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
import torch.nn.functional as F
from tqdm import tqdm
import logging
import os
import argparse

from dinoflow.data import TubeData, compose
from dinoflow.plmodels import BinaryClassificationModel
from dinoflow.models import BTMTubes, munge_state_dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def subsample_events(x, num_events):
    """
    Select a random sample of events and return those.
    If the input has fewer events than `num_events`, all available events are used.
    """
    if len(x.shape) == 3:
        num_available = x.shape[1]
        ev = torch.randperm(num_available)[:min(num_events, num_available)]
        return x[:, ev, :]
    elif len(x.shape) == 2:
        num_available = x.shape[0]
        ev = torch.randperm(num_available)[:min(num_events, num_available)]
        return x[ev, :]
    else:
        raise ValueError(f"Input tensor must have 2 or 3 dimensions (found {len(x.shape)})")

def load_train3tubes_checkpoint(checkpoint_path):
    """
    Load a BinaryClassificationModel with BTMTubes from checkpoint
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    hparams = checkpoint['hyper_parameters']
    state_dict = checkpoint['state_dict']
    
    logger.info(f"Checkpoint hyperparameters: {hparams}")
    
    # Detect layer type from state dict keys
    has_linear_gate = any('linear_gate' in key for key in state_dict.keys())
    layertype = 'swiglu' if has_linear_gate else 'normal'
    logger.info(f"Detected layer type: {layertype}")
    
    # Extract d_ff from the actual tensor shapes in the state dict
    d_ff = None
    for key, tensor in state_dict.items():
        if 'linear1.weight' in key and 'model.' in key:
            d_ff = tensor.shape[0]  # First dimension is d_ff
            break
    
    if d_ff is None:
        d_ff = hparams['model_conf'].get('d_ff', 2048)
    
    logger.info(f"Detected d_ff: {d_ff}")
    
    # Reconstruct the BTMTubes model with correct parameters
    btm_model = BTMTubes(
        num_features=13,
        model_embed_dim=hparams['model_conf']['model_embed_dim'],
        backbone_heads=hparams['model_conf']['backbone_heads'],
        backbone_layers=hparams['model_conf']['backbone_layers'],
        output_classes=hparams['model_conf']['output_classes'],
        d_ff=d_ff,
        layer_type=layertype,
        output_scale_factor=hparams['model_conf'].get('output_scale_factor', 1.0)
    )
    
    # Create the BinaryClassificationModel
    model = BinaryClassificationModel(
        model=btm_model,
        min_lr=hparams.get('min_lr', 1e-5),
        max_lr=hparams.get('max_lr', 0.00025),
        warmup_iters=hparams.get('warmup_iters', 10),
        lr_decay_iters=hparams.get('lr_decay_iters', 80),
        emit_predictions=False,
        ckpt_params=hparams
    )
    
    # Load the state dict
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def compute_plmodels_metrics(y_true, y_pred_probs, fpr_thresholds=[0.01, 0.02, 0.05], sensitivity_thresholds=[0.95, 0.99, 0.995]):
    """
    Compute metrics EXACTLY like BinaryClassificationModel.on_validation_epoch_end
    This replicates the exact same computation used during training
    """
    # Convert to numpy for sklearn compatibility
    gathered_labels = np.array(y_true).astype(int)
    gathered_preds = np.array(y_pred_probs)
    
    metrics = {}
    
    # ROC curve analysis (same as plmodels)
    fpr, tpr, thresholds_roc = roc_curve(gathered_labels, gathered_preds)
    roc_auc = auc(fpr, tpr)
    metrics['roc_auc'] = float(roc_auc)
    
    # Compute sensitivity at each FPR threshold (same as plmodels)
    sensitivities = np.zeros(len(fpr_thresholds))
    for i, fpr_threshold in enumerate(fpr_thresholds):
        idx = fpr <= fpr_threshold
        sensitivity = tpr[idx]
        if len(sensitivity) > 0:
            sensitivities[i] = max(sensitivity)
        else:
            sensitivities[i] = 0
        metrics[f'recall_at_fpr_{fpr_threshold}'] = float(sensitivities[i])
    
    # Compute specificity at each sensitivity threshold (same as plmodels)
    specificities = np.zeros(len(sensitivity_thresholds))
    for i, sens_threshold in enumerate(sensitivity_thresholds):
        idx = tpr >= sens_threshold
        specificity = 1 - fpr[idx]  # Convert FPR to specificity
        if len(specificity) > 0:
            specificities[i] = max(specificity)
        else:
            specificities[i] = 0
        metrics[f'specificity_at_recall_{sens_threshold}'] = float(specificities[i])
    
    # Precision-Recall curve analysis (same as plmodels)
    precision_vals, recall_vals, thresholds_pr = precision_recall_curve(gathered_labels, gathered_preds)
    avg_precision = average_precision_score(gathered_labels, gathered_preds)
    metrics['auprc'] = float(avg_precision)
    
    # Find threshold that maximizes F1 score (same as plmodels)
    f1_scores = 2 * precision_vals * recall_vals / (precision_vals + recall_vals + 1e-8)
    max_f1_idx = np.argmax(f1_scores)
    threshold = thresholds_pr[max_f1_idx] if max_f1_idx < len(thresholds_pr) else 0.5
    best_recall = recall_vals[max_f1_idx]
    best_precision = precision_vals[max_f1_idx]
    best_f1 = f1_scores[max_f1_idx]
    
    metrics['threshold'] = float(threshold)
    metrics['recall'] = float(best_recall)
    metrics['precision'] = float(best_precision)
    metrics['fscore'] = float(best_f1)
    
    # Calculate additional metrics using the SAME approach as plmodels
    # PPV is the same as precision (already computed as best_precision)
    ppv = best_precision
    
    # Apply the F1-maximizing threshold to get binary predictions
    binary_preds = (gathered_preds >= threshold).astype(int)
    
    # Calculate confusion matrix components at the F1-maximizing threshold
    tp = np.sum((binary_preds == 1) & (gathered_labels == 1))
    tn = np.sum((binary_preds == 0) & (gathered_labels == 0))
    fp = np.sum((binary_preds == 1) & (gathered_labels == 0))
    fn = np.sum((binary_preds == 0) & (gathered_labels == 1))
    
    # Calculate derived metrics at the same threshold (same as plmodels)
    specificity_val = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
    accuracy_f1_threshold = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    # Add these metrics (same names as logged in plmodels)
    metrics['specificity'] = float(specificity_val)
    metrics['ppv'] = float(ppv)
    metrics['npv'] = float(npv)
    metrics['accuracy'] = float(accuracy_f1_threshold)
    
    # Add confusion matrix components for reference
    metrics['true_positives'] = int(tp)
    metrics['false_positives'] = int(fp)
    metrics['true_negatives'] = int(tn)
    metrics['false_negatives'] = int(fn)
    
    return metrics

def evaluate_with_plmetrics(
    model,
    dataset_csv: str,
    dataroot: str,
    labelkey: str,
    events_per_sample: int = 4096,
    num_subsamples: int = 1,
    batch_size: int = 8,
    device: str = None,
    focus_accessions: list = None
):
    """
    Evaluate dataset and return both individual predictions and PLModels-style metrics
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load dataset (without fixed subsampling)
    dataset = TubeData(
        dataset_csv,
        events_to_return=-1,  # Don't subsample yet
        data_root=dataroot,
        tubes_to_return=["b", "t", "m"],
        labelkey=labelkey,
        transforms=None
    )
    
    predictions = []
    
    # If focusing on specific accessions
    if focus_accessions:
        indices = [i for i in range(len(dataset)) if dataset.get_row_data(i)['ACCESSION'] in focus_accessions]
    else:
        indices = range(len(dataset))
    
    for idx in tqdm(indices, desc="Processing samples"):
        row_data = dataset.get_row_data(idx)
        accession = row_data['ACCESSION']
        true_label = row_data[labelkey]
        
        # Get the full tube data (not subsampled yet)
        full_tubes, _ = dataset[idx]
        
        # Skip if any tube has fewer events than `events_per_sample`
        skip_sample = False
        for tube, tube_data in full_tubes.items():
            if len(tube_data.shape) == 3:
                num_events = tube_data.shape[1]
            elif len(tube_data.shape) == 2:
                num_events = tube_data.shape[0]
            else:
                raise ValueError(f"Invalid shape for tube {tube}: {tube_data.shape}")
            
            if num_events < events_per_sample:
                skip_sample = True
                break
        
        if skip_sample:
            continue  # Skip this sample
        
        # Run multiple subsamples
        subsample_logits = []
        subsample_probs = []
        
        for subsample_idx in range(num_subsamples):
            # Manually subsample the tubes
            subsampled_tubes = {
                tube: subsample_events(tube_data, events_per_sample)
                for tube, tube_data in full_tubes.items()
            }
            
            # Convert to batch format (batch size 1)
            batch = {tube: data.unsqueeze(0).to(device) for tube, data in subsampled_tubes.items()}
            
            # Predict using autocast like in training
            with torch.no_grad(), torch.autocast(device_type='cuda' if 'cuda' in str(device) else 'cpu', dtype=torch.bfloat16):
                logits = model(batch)
                prob = torch.sigmoid(logits.squeeze(1)).float().item()  # Convert to float32
                logit_val = logits.squeeze(1).float().item()
                
                subsample_logits.append(logit_val)
                subsample_probs.append(prob)
        
        # Store results for this sample
        if num_subsamples == 1:
            predictions.append({
                'accession': accession,
                'true_label': true_label,
                'logit': subsample_logits[0],
                'probability': subsample_probs[0]
            })
        else:
            # Multiple subsamples - store statistics
            predictions.append({
                'accession': accession,
                'true_label': true_label,
                'mean_logit': np.mean(subsample_logits),
                'std_logit': np.std(subsample_logits),
                'mean_probability': np.mean(subsample_probs),
                'std_probability': np.std(subsample_probs),
                'all_logits': subsample_logits,
                'all_probabilities': subsample_probs
            })
    
    predictions_df = pd.DataFrame(predictions)
    
    # Compute PLModels-style metrics
    if num_subsamples == 1:
        y_true = predictions_df['true_label'].values
        y_pred_probs = predictions_df['probability'].values
    else:
        y_true = predictions_df['true_label'].values
        y_pred_probs = predictions_df['mean_probability'].values
    
    metrics = compute_plmodels_metrics(y_true, y_pred_probs)
    
    return predictions_df, metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate train3tubes model with PLModels-style metrics")
    parser.add_argument("checkpoint_path", help="Path to the checkpoint file")
    parser.add_argument("dataset_csv", help="Path to the dataset CSV file")
    parser.add_argument("--dataroot", default=".", help="Root directory for the data")
    parser.add_argument("--labelkey", default="label", help="Label column name in CSV")
    parser.add_argument("--events", type=int, default=4096, help="Number of events per sample")
    parser.add_argument("--num-subsamples", type=int, default=1, help="Number of subsamples per sample")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--output-prefix", required=True, help="Prefix for output files")
    parser.add_argument("--device", default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--focus-accessions", nargs="+", help="Only evaluate specific accessions")
    
    args = parser.parse_args()
    
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    logger.info(f"Loading model from: {args.checkpoint_path}")
    
    # Load model
    model = load_train3tubes_checkpoint(args.checkpoint_path)
    model.eval()
    model.to(device)
    
    # Run evaluation
    logger.info(f"Evaluating on dataset: {args.dataset_csv}")
    logger.info(f"Events per sample: {args.events}, Number of subsamples: {args.num_subsamples}")
    
    predictions_df, metrics = evaluate_with_plmetrics(
        model=model,
        dataset_csv=args.dataset_csv,
        dataroot=args.dataroot,
        labelkey=args.labelkey,
        events_per_sample=args.events,
        num_subsamples=args.num_subsamples,
        batch_size=args.batch_size,
        device=device,
        focus_accessions=args.focus_accessions
    )
    
    if len(predictions_df) == 0:
        logger.error("No samples could be processed! Check your data and event requirements.")
        return
    
    # Save predictions
    predictions_csv = f"{args.output_prefix}_predictions.csv"
    predictions_df.to_csv(predictions_csv, index=False)
    logger.info(f"Predictions saved to: {predictions_csv}")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_csv = f"{args.output_prefix}_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    logger.info(f"Metrics saved to: {metrics_csv}")
    
    # Print results
    logger.info("=" * 60)
    logger.info("TRAIN3TUBES EVALUATION WITH PLMODELS METRICS")
    logger.info("=" * 60)
    
    y_true = predictions_df['true_label'].values
    if args.num_subsamples == 1:
        y_pred_probs = predictions_df['probability'].values
    else:
        y_pred_probs = predictions_df['mean_probability'].values
    
    print(f"Dataset: {args.dataset_csv}")
    print(f"Number of samples processed: {len(predictions_df)}")
    print(f"Events per sample: {args.events}")
    print(f"Number of subsamples: {args.num_subsamples}")
    print(f"Positive samples: {np.sum(y_true)} ({100*np.mean(y_true):.1f}%)")
    print(f"Negative samples: {len(y_true) - np.sum(y_true)} ({100*(1-np.mean(y_true)):.1f}%)")
    print()
    
    print("PLMODELS-STYLE METRICS (F1-maximizing threshold):")
    print(f"  Accuracy: {metrics['accuracy']:.6f}")
    print(f"  Precision: {metrics['precision']:.6f}")
    print(f"  Recall: {metrics['recall']:.6f}")
    print(f"  F1-Score: {metrics['fscore']:.6f}")
    print(f"  Specificity: {metrics['specificity']:.6f}")
    print(f"  PPV: {metrics['ppv']:.6f}")
    print(f"  NPV: {metrics['npv']:.6f}")
    print(f"  F1-Maximizing Threshold: {metrics['threshold']:.6f}")
    print(f"  ROC AUC: {metrics['roc_auc']:.6f}")
    print(f"  AUPRC: {metrics['auprc']:.6f}")
    print()
    
    print("SENSITIVITY AT FPR THRESHOLDS:")
    for fpr_thresh in [0.01, 0.02, 0.05]:
        recall_val = metrics[f'recall_at_fpr_{fpr_thresh}']
        print(f"  recall_at_fpr_{fpr_thresh}: {recall_val:.6f}")
    print()
    
    print("SPECIFICITY AT RECALL THRESHOLDS:")
    for sens_thresh in [0.95, 0.99, 0.995]:
        spec_val = metrics[f'specificity_at_recall_{sens_thresh}']
        print(f"  specificity_at_recall_{sens_thresh}: {spec_val:.6f}")
    print()
    
    print("CONFUSION MATRIX (F1-maximizing threshold):")
    print(f"  True Positives: {metrics['true_positives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  True Negatives: {metrics['true_negatives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print()
    
    if args.num_subsamples > 1:
        print("SUBSAMPLING STATISTICS:")
        print(f"  Mean std of probabilities: {predictions_df['std_probability'].mean():.6f}")
        print(f"  Max std of probabilities: {predictions_df['std_probability'].max():.6f}")

if __name__ == "__main__":
    main()


#Usage Example:
    # python dinoflow/scripts/evaluate_with_plmetrics.py \
    # /home/32210/git/dinoflow/dinoflow_eval_dinoflow_ar_16kev_PB_lr0001/last.ckpt \
    # /home/32210/test_things/lmd_conversion/updated_test_set_09052025.csv \
    # --dataroot /mnt/ri_share/Data/flow_data/ \
    # --labelkey ACTION_REQUIRED \
    # --events 16384 \
    # --output-prefix my_evaluation \
    # --num-subsamples 3