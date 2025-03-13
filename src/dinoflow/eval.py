import logging
from functools import partial

import typer
import yaml
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import BinaryPrecisionRecallCurve, BinaryF1Score, BinaryRecall, BinaryPrecision, BinaryAccuracy
from torchmetrics.aggregation import MeanMetric, SumMetric
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CometLogger
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score

from torch.utils.data import DataLoader, Dataset
import numpy as np


from dinoflow.models import TubeEncoder, TubeEncoderWithProjection, load_checkpoint, BTMTubes
from dinoflow.data import TubeData, collate_fn, compose, shift, scale, noise, standardize_range
from dinoflow import util


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = typer.Typer(pretty_exceptions_show_locals=False)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]   %(levelname)s   %(message)s')

logger = logging.getLogger(__name__)


class ClassificationHead(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.GELU(),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x):
        return self.layers(x)
    

class CombinedModel(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
    
    def forward(self, x):
        return self.classifier(self.backbone(x.float()))


class BinaryClassificationModel(pl.LightningModule):
    def __init__(self, model, min_lr=0.00001, max_lr=0.0001, warmup_iters=20, lr_decay_iters=250, emit_predictions=False, ckpt_params=None):
        super().__init__()
        self.model = model #
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.accuracy = BinaryAccuracy()
        self.training_loss_mean = MeanMetric()
        self.validation_loss_mean = MeanMetric()
        self.emit_predictions = emit_predictions
        if ckpt_params is not None:
            self.save_hyperparameters(ckpt_params)
        
        # Lists to collect predictions and labels
        self.val_preds = []
        self.val_labels = []

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, rowinfo = batch
        labels = rowinfo['label']
        preds = self(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(preds.squeeze(1), labels.float())
        self.training_loss_mean.update(loss)
        return loss
    
    def on_validation_epoch_start(self):
        # Clear the lists at the start of validation
        self.val_preds = []
        self.val_labels = []
    
    def validation_step(self, batch, batch_idx):
        x, rowinfo = batch
        labels = rowinfo['label']
        accs = rowinfo['ACCESSION']
        preds = self(x).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(preds, labels.float())
        preds = torch.nn.Sigmoid()(preds) # raw outputs are logits, non-sigmoid
        
        # Store predictions and labels for later use
        self.val_preds.append(preds.detach().float())
        self.val_labels.append(labels.detach().float())
        
        if self.emit_predictions:
            for p, l, a in zip(preds, labels, accs):
                if l == 1 or p.item() > 0.5:
                    print(f"{a}\t{p.item() :.4f}\t{l.item() :.2f}")
        self.accuracy(preds, labels)
        
        self.validation_loss_mean.update(loss)

    def on_validation_epoch_end(self):
        lrsched = self.lr_schedulers()
        lr = lrsched.get_last_lr()[0]
        accuracy = self.accuracy.compute()
        
        # Gather predictions and labels from all processes
        if self.trainer.world_size > 1:
            # For distributed training
            gathered_preds = self.all_gather(torch.cat(self.val_preds).float()).flatten()
            gathered_labels = self.all_gather(torch.cat(self.val_labels).int()).flatten()
            
            # Reshape if needed
            if gathered_preds.dim() > 2:
                gathered_preds = gathered_preds.reshape(-1)
            if gathered_labels.dim() > 2:
                gathered_labels = gathered_labels.reshape(-1)
        else:
            # For single process
            gathered_preds = torch.cat(self.val_preds).float()
            gathered_labels = torch.cat(self.val_labels).int()
 
        # These get set in the main process, but for logging we are required to do that in every process, so set
        # some defaults here for every process ...
        best_f1 = float("NaN")
        best_precision = float("NaN")
        threshold = float("NaN")
        best_recall = float("NaN")
        # Only create and log the plot on the main process
        if self.trainer.is_global_zero and isinstance(self.logger, CometLogger):
           
            fpr, tpr, _ = roc_curve(gathered_labels.cpu().numpy(), gathered_preds.cpu().numpy())
            roc_auc = auc(fpr, tpr)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic')
            ax.legend(loc="lower right")
            ax.grid(True)
            
            # Log the ROC curve to CometML
            self.logger.experiment.log_figure(figure=fig, figure_name="ROC_Curve", step=self.current_epoch)
            plt.close(fig)
            
            # Create Precision-Recall curve
            precision_vals, recall_vals, thresholds = precision_recall_curve(gathered_labels.cpu().numpy(), gathered_preds.cpu().numpy())
            avg_precision = average_precision_score(gathered_labels.cpu().numpy(), gathered_preds.cpu().numpy())
            
            # Find threshold that maximizes F1 score
            f1_scores = 2 * precision_vals * recall_vals / (precision_vals + recall_vals)
            max_f1_idx = np.argmax(f1_scores)
            threshold = thresholds[max_f1_idx]
            best_recall = recall_vals[max_f1_idx]
            best_precision = precision_vals[max_f1_idx]
            best_f1 = f1_scores[max_f1_idx]

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(recall_vals, precision_vals, label=f'PR curve (AP = {avg_precision:.3f})')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            ax.legend(loc="lower left")
            ax.grid(True)
            
            # Log the PR curve to CometML
            self.logger.experiment.log_figure(figure=fig, figure_name="PR_Curve", step=self.current_epoch)
            plt.close(fig)

        # We only want to log these from the main process (where self.trainer.is_global_zero is true), and it will hang
        # if we have sync_dict=True since we are not syncing anything across processes here
        self.log('recall', best_recall)
        self.log('fscore', best_f1)
        self.log('threshold', threshold)
        self.log('precision', best_precision)

        # Process syncing is handled by lightning for these, and we want sync_dist=True to make sure things are synced across processes 
        self.log('accuracy', accuracy, sync_dist=True)
        self.log('val_loss', self.validation_loss_mean.compute(), sync_dist=True)
        self.log('training_loss', self.training_loss_mean.compute(), sync_dist=True)
        self.log('learning_rate', lr)
    
        self.accuracy.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        lrschedule = util.WarmupCosineLRScheduler(optimizer, self.max_lr, self.min_lr, self.warmup_iters, self.lr_decay_iters)
        return [optimizer], [lrschedule]

    # Add these methods to specify checkpoint monitoring preferences
    @property
    def checkpoint_monitor(self):
        return 'fscore'
    
    @property
    def checkpoint_mode(self):
        return 'max'
        
    @property
    def comet_project(self):
        return 'dinoflow-classifier'


class ClassificationModel(pl.LightningModule):
    def __init__(self, model, num_classes, min_lr=0.00001, max_lr=0.0001, warmup_iters=20, lr_decay_iters=250, emit_predictions=False, ckpt_params=None):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        
        # Multi-class metrics
        from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
        self.accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.f1_score = MulticlassF1Score(num_classes=num_classes)
        
        self.training_loss_mean = MeanMetric()
        self.validation_loss_mean = MeanMetric()
        self.emit_predictions = emit_predictions
        if ckpt_params is not None:
            self.save_hyperparameters(ckpt_params)
        
        # Lists to collect predictions and labels
        self.val_preds = []
        self.val_labels = []

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, rowinfo = batch
        labels = rowinfo['label']
        preds = self(x)
        loss = torch.nn.functional.cross_entropy(preds, labels.long())
        self.training_loss_mean.update(loss)
        return loss
    
    def on_validation_epoch_start(self):
        # Clear the lists at the start of validation
        self.val_preds = []
        self.val_labels = []
    
    def validation_step(self, batch, batch_idx):
        x, rowinfo = batch
        labels = rowinfo['label']
        accs = rowinfo['ACCESSION']
        preds = self(x)
        loss = torch.nn.functional.cross_entropy(preds, labels.long())
        
        # Get class predictions
        pred_classes = torch.argmax(preds, dim=1)
        
        # Store predictions and labels for later use
        self.val_preds.append(pred_classes.detach())
        self.val_labels.append(labels.detach())
        
        if self.emit_predictions:
            for p, l, a in zip(pred_classes, labels, accs):
                print(f"{a}\t{p.item()}\t{l.item()}")
        
        # Update metrics
        self.accuracy(preds, labels.long())
        self.f1_score(preds, labels.long())
        self.validation_loss_mean.update(loss)

    def on_validation_epoch_end(self):
        lrsched = self.lr_schedulers()
        lr = lrsched.get_last_lr()[0]
        accuracy = self.accuracy.compute()
        f1 = self.f1_score.compute()
        
        # Gather predictions and labels from all processes
        if self.trainer.world_size > 1:
            # For distributed training
            gathered_preds = self.all_gather(torch.cat(self.val_preds))
            gathered_labels = self.all_gather(torch.cat(self.val_labels))
            
            # Reshape if needed
            if gathered_preds.dim() > 1 and gathered_preds.size(0) == self.trainer.world_size:
                gathered_preds = gathered_preds.view(-1)
            if gathered_labels.dim() > 1 and gathered_labels.size(0) == self.trainer.world_size:
                gathered_labels = gathered_labels.view(-1)
        else:
            # For single process
            gathered_preds = torch.cat(self.val_preds)
            gathered_labels = torch.cat(self.val_labels)
        
        # Only create and log the confusion matrix on the main process
        if self.trainer.is_global_zero and isinstance(self.logger, CometLogger):
            from sklearn.metrics import confusion_matrix
            import seaborn as sns
            
            # Create confusion matrix
            cm = confusion_matrix(
                gathered_labels.cpu().numpy(), 
                gathered_preds.cpu().numpy(),
                labels=range(self.num_classes)
            )
            
            # Plot confusion matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            ax.set_title('Confusion Matrix')
            
            # If we have more than 10 classes, don't show all tick labels
            if self.num_classes <= 10:
                ax.set_xticks(np.arange(self.num_classes) + 0.5)
                ax.set_yticks(np.arange(self.num_classes) + 0.5)
                ax.set_xticklabels(range(self.num_classes))
                ax.set_yticklabels(range(self.num_classes))
            
            # Log the confusion matrix to CometML
            self.logger.experiment.log_figure(figure=fig, figure_name="Confusion_Matrix", step=self.current_epoch)
            plt.close(fig)

        # Log metrics
        self.log('accuracy', accuracy, sync_dist=True)
        self.log('f1_score', f1, sync_dist=True)
        self.log('val_loss', self.validation_loss_mean.compute(), sync_dist=True)
        self.log('training_loss', self.training_loss_mean.compute(), sync_dist=True)
        self.log('learning_rate', lr)
        
        # Reset metrics
        self.accuracy.reset()
        self.f1_score.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        lrschedule = util.WarmupCosineLRScheduler(optimizer, self.max_lr, self.min_lr, self.warmup_iters, self.lr_decay_iters)
        return [optimizer], [lrschedule]

    # Add these methods to specify checkpoint monitoring preferences
    @property
    def checkpoint_monitor(self):
        return 'f1_score'
    
    @property
    def checkpoint_mode(self):
        return 'max'
        
    @property
    def comet_project(self):
        return 'dinoflow-classifier'


class RegressionModel(pl.LightningModule):
    def __init__(self, model, min_lr=0.00001, max_lr=0.0001, warmup_iters=20, lr_decay_iters=250, emit_predictions=False, ckpt_params=None):
        super().__init__()
        self.model = model
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.emit_predictions = emit_predictions
        if ckpt_params is not None:
            self.save_hyperparameters(ckpt_params)
        # Regression metrics
        self.mse = MeanSquaredError()
        self.rmse = MeanSquaredError(squared=False)  # RMSE is sqrt of MSE
        self.mae = MeanAbsoluteError()
        self.r2 = R2Score()
        
        self.training_loss_mean = MeanMetric()
        self.validation_loss_mean = MeanMetric()
        
        # Lists to collect predictions and labels
        self.val_preds = []
        self.val_labels = []

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, rowinfo = batch
        labels = rowinfo['label']
        preds = F.sigmoid(self(x).squeeze(1)) * 100.0
        loss = F.mse_loss(preds, labels.float())
        self.training_loss_mean.update(loss)
        return loss
    
    def on_validation_epoch_start(self):
        # Clear the lists at the start of validation
        self.val_preds = []
        self.val_labels = []
    
    def validation_step(self, batch, batch_idx):
        x, rowinfo = batch
        labels = rowinfo['label']
        accs = rowinfo['ACCESSION']
        preds = F.sigmoid(self(x).squeeze(1)) * 100.0
        loss = F.mse_loss(preds, labels.float())
        
        # Store predictions and labels for later use
        self.val_preds.append(preds.detach())
        self.val_labels.append(labels.detach())
        
        if self.emit_predictions:
            for p, l, a in zip(preds, labels, accs):
                print(f"{a}\t{p.item():.4f}\t{l.item():.4f}")
        
        # Update metrics
        self.mse(preds, labels)
        self.rmse(preds, labels)
        self.mae(preds, labels)
        self.r2(preds, labels)
        self.validation_loss_mean.update(loss)

    def on_validation_epoch_end(self):
        lrsched = self.lr_schedulers()
        lr = lrsched.get_last_lr()[0]
        
        # Compute metrics
        mse = self.mse.compute()
        rmse = self.rmse.compute()
        mae = self.mae.compute()
        r2 = self.r2.compute()
        
        # Gather predictions and labels from all processes
        if self.trainer.world_size > 1:
            # For distributed training
            gathered_preds = self.all_gather(torch.cat(self.val_preds).float())
            gathered_labels = self.all_gather(torch.cat(self.val_labels).float())
            
            # Reshape if needed
            if gathered_preds.dim() > 2:
                gathered_preds = gathered_preds.reshape(-1)
            if gathered_labels.dim() > 2:
                gathered_labels = gathered_labels.reshape(-1)
        else:
            # For single process
            gathered_preds = torch.cat(self.val_preds).float()
            gathered_labels = torch.cat(self.val_labels).float()
        
        # Only create and log the plot on the main process
        if self.trainer.is_global_zero:
            # Create scatter plot of predictions vs actual values
            if isinstance(self.logger, CometLogger):
                import matplotlib.pyplot as plt
                
                # Create scatter plot
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.scatter(gathered_labels.cpu().numpy(), gathered_preds.cpu().numpy(), alpha=0.5)
                
                # Add perfect prediction line
                min_val = min(gathered_labels.min().item(), gathered_preds.min().item())
                max_val = max(gathered_labels.max().item(), gathered_preds.max().item())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--')
                
                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Predicted Values')
                ax.set_title(f'Predictions vs Actual (R² = {r2:.3f}, RMSE = {rmse:.3f})')
                ax.grid(True)
                
                # Log the figure to CometML
                self.logger.experiment.log_figure(figure=fig, figure_name="Predictions_vs_Actual", step=self.current_epoch)
                plt.close(fig)
        
        # Log metrics
        self.log('mse', mse, sync_dist=True)
        self.log('rmse', rmse, sync_dist=True)
        self.log('mae', mae, sync_dist=True)
        self.log('r2', r2, sync_dist=True)
        self.log('val_loss', self.validation_loss_mean.compute(), sync_dist=True)
        self.log('training_loss', self.training_loss_mean.compute(), sync_dist=True)
        self.log('learning_rate', lr)
        
        # Reset metrics
        self.mse.reset()
        self.rmse.reset()
        self.mae.reset()
        self.r2.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        lrschedule = util.WarmupCosineLRScheduler(optimizer, self.max_lr, self.min_lr, self.warmup_iters, self.lr_decay_iters)
        return [optimizer], [lrschedule]

    # Add these methods to specify checkpoint monitoring preferences
    @property
    def checkpoint_monitor(self):
        return 'rmse'
    
    @property
    def checkpoint_mode(self):
        return 'min'
        
    @property
    def comet_project(self):
        return 'dinoflow-viability'


def load_featmeans_stds(conf, tube_type):
    if tube_type == 't':
        return conf['normalization_params']['t_feat_means'], conf['normalization_params']['t_feat_stds']
    elif tube_type == 'm':
        return conf['normalization_params']['m_feat_means'], conf['normalization_params']['m_feat_stds']
    elif tube_type == 'b':
        return conf['normalization_params']['b_feat_means'], conf['normalization_params']['b_feat_stds']
    else:
        raise ValueError(f"Unknown tube type: {tube_type}")


def _run_trainer(model, train_labels, test_labels, tubes, run_name, labelkey, dataroot, events, batch_size, epochs, positive_repeat_factor=1):
    torch.set_float32_matmul_precision('medium')

    # Repeat rows in positive samples to balance the dataset
    if positive_repeat_factor > 1:
        train_labels = pd.read_csv(train_labels)
        positive_rows = train_labels[train_labels[labelkey] == 1]
        logger.info(f"Positive samples: {len(positive_rows)}")
        new_rows =  pd.concat([positive_rows] * positive_repeat_factor, ignore_index=True)
        # Concatenate the repeated rows back to the original DataFrame
        train_labels = pd.concat([train_labels, new_rows], ignore_index=True)
        logger.info(f"New positive samples: {len(train_labels[train_labels[labelkey] == 1])}")

    train_transforms = compose([
        partial(shift, scale=0.1),
        partial(scale, scale=0.1),
        partial(noise, scale=0.25),
    ])

    val_transforms = compose([
    ])

    # Use the model's specified checkpoint monitor values instead of hardcoding them
    checkpoint_monitor_val = model.checkpoint_monitor
    checkpoint_monitor_mode = model.checkpoint_mode
    comet_project = model.comet_project

    traindata = TubeData(train_labels, tubes_to_return=tubes, events_to_return=int(events), data_root=dataroot, labelkey=labelkey, transforms=train_transforms)
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=8)
    logger.info(f"Loaded {len(trainloader.dataset)} samples for training")
    
    # Check if we have any positive samples for classification models
    if hasattr(traindata, 'positive_negative_samples'):
        logger.info(f"Positive samples: {len(traindata.positive_negative_samples()[0])}")
        logger.info(f"Negative samples: {len(traindata.positive_negative_samples()[1])}")
        assert len(traindata.positive_negative_samples()[0]) > 0, f"No positive samples found :("
    
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=16)    
    logger.info(f"Loaded {len(trainloader.dataset)} samples for training")

    valdata = TubeData(test_labels, tubes_to_return=tubes, events_to_return=int(events), data_root=dataroot, labelkey=labelkey, transforms=val_transforms)
    valloader = DataLoader(valdata, batch_size=batch_size, shuffle=False, num_workers=8)
    logger.info(f"Loaded {len(valloader.dataset)} samples for val")
    
    # Check if we have any positive samples for classification models in the validation set
    if hasattr(valdata, 'positive_negative_samples'):
        logger.info(f"Positive samples: {len(valdata.positive_negative_samples()[0])}")
        logger.info(f"Negative samples: {len(valdata.positive_negative_samples()[1])}")
        #assert len(valdata.positive_negative_samples()[0]) > 0, f"No positive samples found :("

    valloader = DataLoader(valdata, batch_size=batch_size, shuffle=False, num_workers=16)
    logger.info(f"Loaded {len(valloader.dataset)} samples for val")

    comet_logger = CometLogger(
            workspace="brendan",  # Optional
            save_dir="dinoflow_classifier_runs",  # Optional
            project_name=comet_project,  # Optional
            experiment_name=run_name,  # Optional
        )

    trainer = pl.Trainer(max_epochs=epochs,
                        accelerator='auto',
                        precision="bf16-mixed",
                        callbacks=[
                            ModelCheckpoint(dirpath=f"dinoflow_eval_{run_name}", monitor=checkpoint_monitor_val, mode=checkpoint_monitor_mode, save_top_k=1, save_last=True, filename=run_name + "_e{epoch}"),
                            LearningRateMonitor(logging_interval='step'),
                        ],
                        logger=comet_logger)

    trainer.fit(model, trainloader, valloader)


def munge_state_dict(state_dict):
    """
    Required to load the state dict from a checkpoint into a new model
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('model.', '')
        new_state_dict[new_key] = value
    return new_state_dict

@app.command()
def train(run_name, train_labels, test_labels, backbone: str, conf: str, tube_type: str = "", dataroot: str = "/", positive_repeat_factor: int = 1, labelkey: str = "label", checkpoint: str = None, freeze_backbone: bool = False, batch_size: int=16, events: int = 4096, epochs: int = 25, mode: str = 'binary', num_classes: int = 2) :
    """
    Evaluate the model on the test set
    """
    assert tube_type != "", "Tube type must be specified"
    # Helps with too many open files errors?
    torch.multiprocessing.set_sharing_strategy('file_system')

    logger.info(f"Loading backbone from {backbone}")
    backbone, modelconf = load_checkpoint(backbone)
    classifier = ClassificationHead(backbone.cls_token.shape[-1], 1 if mode == 'binary' or mode == 'regression' else num_classes)
    combined = CombinedModel(backbone, classifier)

    if mode == 'binary':
        model = BinaryClassificationModel(combined, emit_predictions=True)
    elif mode == 'multiclass':
        model = ClassificationModel(combined, num_classes=num_classes, emit_predictions=True)
    elif mode == 'regression':
        model = RegressionModel(combined, emit_predictions=True)
        assert positive_repeat_factor == 1
    else:
        raise ValueError(f"Unknown mode: {mode}")

    with open(conf, 'r') as f:
        conf = yaml.safe_load(f)

    # feat_means, feat_stds = load_featmeans_stds(conf, tube_type)
    # feat_means = torch.tensor(feat_means).to(model.device)
    # feat_stds = torch.tensor(feat_stds).to(model.device)

    if checkpoint is not None:
        logger.info(f"Loading full model checkpoint from {checkpoint}")
        if mode == 'binary':
            model = BinaryClassificationModel.load_from_checkpoint(checkpoint, backbone=backbone, classifier=classifier)
        elif mode == 'multiclass':
            model = ClassificationModel.load_from_checkpoint(checkpoint, backbone=backbone, classifier=classifier, num_classes=num_classes)
        elif mode == 'regression':
            model = RegressionModel.load_from_checkpoint(checkpoint, backbone=backbone, classifier=classifier)
    
    if freeze_backbone:
        logger.info("Freezing backbone")
        backbone.eval() # freeze backbone
        for p in backbone.parameters():
            p.requires_grad = False
    else:
        logger.info("Unfreezing backbone")
        backbone.train()

    _run_trainer(model, train_labels, test_labels, [tube_type], run_name, labelkey, dataroot, events, batch_size, epochs, positive_repeat_factor)
    

@app.command()
def train3tubes(b_ckpt, t_ckpt, m_ckpt, 
                train_labels, test_labels,
                run_name,
                labelkey: str = "label",
                dataroot: str = ".",
                events: int = 4096,
                batch_size: int = 16,
                epochs: int = 50,
                mode:str = 'binary',
                positive_repeat_factor: int = 1,
                num_classes: int = 2,
                train_backbone: bool = False):
    # Helps with too many open files errors?
    torch.multiprocessing.set_sharing_strategy('file_system')

    b_backbone, modelconf = load_checkpoint(b_ckpt)
    t_backbone, _ = load_checkpoint(t_ckpt)
    m_backbone, _ = load_checkpoint(m_ckpt)

    # Turn off gradients and set to eval mode
    if not train_backbone:
        logger.info("Freezing backbones")
        b_backbone.eval()
        t_backbone.eval()
        m_backbone.eval()
        for p in b_backbone.parameters():
            p.requires_grad = False
        for p in t_backbone.parameters():
            p.requires_grad = False
        for p in m_backbone.parameters():
            p.requires_grad = False
    else:
        logger.info("Unfreezing backbones")
        b_backbone.train()
        t_backbone.train()
        m_backbone.train()

    output_classes = 1 if mode == 'binary' or mode == 'regression' else num_classes
    modelconf['output_classes'] = output_classes # Add it here so it can be saved in the checkpoint

    btm = BTMTubes(num_features=13,
                    model_embed_dim=modelconf['model_dim'],
                    backbone_heads=modelconf['heads'],
                    backbone_layers=modelconf['layers'],
                    output_classes=output_classes)

    btm.b_backbone = b_backbone
    btm.t_backbone = t_backbone
    btm.m_backbone = m_backbone
    
    if mode == 'binary':
        model = BinaryClassificationModel(btm, emit_predictions=False, ckpt_params=modelconf)
    elif mode == 'multiclass':
        model = ClassificationModel(btm, num_classes=num_classes, emit_predictions=False, ckpt_params=modelconf)
    elif mode == 'regression':
        model = RegressionModel(btm, emit_predictions=False, ckpt_params=modelconf)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    _run_trainer(model, train_labels, test_labels, ["b", "t", "m"], run_name, labelkey, dataroot, events, batch_size, epochs, positive_repeat_factor)
    
    

@app.command()
def predict(checkpoint: str,
            test_labels: str,
            labelkey:str, 
            dataroot: str = ".", 
            events: int = 4096, 
            batch_size: int = 16):
    """
    Predict the labels for the test set
    """
    # In the future we'll be able to load the modelconf from the checkpoint but older models dont save it
    
    ckpt = torch.load(checkpoint, weights_only=False)
    if 'hyper_parameters' in ckpt:
        modelconf = ckpt['hyper_parameters']
    else:
        modelconf = {
            'd_ff': 2048,
            'model_dim': 512,
            'heads': 4,
            'layers': 10,
        }
    ckpt['state_dict'] = munge_state_dict(ckpt['state_dict'])
    if 'num_classes' not in modelconf:
        logger.info(f"num_classes not found in conf, trying to get it from model state dict..")
        bs = ckpt['state_dict']['combined.4.bias'].shape
        logger.info(f"Model final layer shape: {bs}")
        num_classes = ckpt['state_dict']['combined.4.bias'].shape[0]
    else:
        num_classes = modelconf['num_classes']
    logger.info(f"Output classes: {num_classes}")
    model = BTMTubes(num_features=13,
                    model_embed_dim=modelconf['model_dim'],
                    backbone_heads=modelconf['heads'],
                    backbone_layers=modelconf['layers'],
                    d_ff=modelconf.get('d_ff', 2048),
                    output_classes=num_classes)
    model.load_state_dict(ckpt['state_dict'])
    model.eval().to(DEVICE)
    
    testdata = TubeData(test_labels, data_root=dataroot, labelkey=labelkey, tubes_to_return=["b", "t", "m"], events_to_return=int(events))
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=4)
    logger.info(f"Loaded {len(testloader.dataset)} samples for test")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.inference_mode():
        print("index,accession,prediction,label")
        for b, (batch, rowdict) in enumerate(testloader):
            labels = rowdict['label']
            i = 0
            batch = {k : batch[k].to(DEVICE) for k in ["b", "t", "m"]}
            preds = model(batch)
            if num_classes == 1:
                preds = F.sigmoid(preds)
            else:
                preds = F.softmax(preds, dim=1)
            for p, l in zip(preds, labels):
                idx = b * batch_size + i
                if num_classes == 1:
                    p = f"{p.item() :.4f}"
                else:
                    p = p.argmax(dim=0).item()
                print(f"{idx},{rowdict['ACCESSION'][i]},{p},{labels[i]}")
                i += 1

if __name__ == "__main__":
    app()
