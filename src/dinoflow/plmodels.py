import torch
import pytorch_lightning as pl
from torchmetrics import MeanMetric, MeanSquaredError, MeanAbsoluteError, R2Score, ConfusionMatrix
from torchmetrics.classification import BinaryAccuracy, BinarySpecificity, BinaryPrecision, BinaryRecall
from torchmetrics.aggregation import MeanMetric, SumMetric
import torch.nn.functional as F

from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.utilities import rank_zero_only
from dinoflow import util
from dinoflow.loss import InfoNCELoss, BinarySupervisedMultimodalContrastiveLoss

class BinaryClassificationModel(pl.LightningModule):
    def __init__(self, model, min_lr=0.00001, max_lr=0.00025, warmup_iters=10, lr_decay_iters=80, emit_predictions=False, ckpt_params=None, num_classes=1, comet_project_name=None, freeze_encoder_iters=0, pos_weight=1.0, checkpoint_monitor='specificity_at_recall_0.99', checkpoint_mode='max'):
        super().__init__()
        assert num_classes==1, "Only one class permitted for binary"
        self.model = model #
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.freeze_encoder_iters = freeze_encoder_iters
        self.pos_weight = pos_weight
        self.checkpoint_monitor_metric = checkpoint_monitor
        self.checkpoint_monitor_mode = checkpoint_mode
        self.accuracy = BinaryAccuracy()
        self.specificity = BinarySpecificity()
        self.precision = BinaryPrecision()  # This is PPV
        self.recall = BinaryRecall()        # For NPV calculation
        self.confusion_matrix = ConfusionMatrix(task='binary')
        self.training_loss_mean = MeanMetric()
        self.validation_loss_mean = MeanMetric()
        self.emit_predictions = emit_predictions
        self.comet_project_name = comet_project_name
        # These are the thresholds at which we compute the sensitivity
        # Probably best not to change them
        self.fpr_thresholds = [0.01, 0.02, 0.05]
        # Sensitivity thresholds for computing specificity (equivalent to FNR thresholds)
        # FNR = 1 - Sensitivity, so sensitivity 0.99 = FNR 0.01
        self.sensitivity_thresholds = [0.95, 0.99, 0.995]
        if ckpt_params is None:
            ckpt_params = {}
        ckpt_params['model_class'] = self.__class__.__name__
        ckpt_params['backbone_class'] = model.__class__.__name__
        if hasattr(model, 'model_conf'):
            ckpt_params['model_conf'] = model.model_conf
        else:
            ckpt_params['model_conf'] = {}
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
        pos_weight_tensor = torch.tensor([self.pos_weight], device=preds.device)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            preds.squeeze(1), 
            labels.float(), 
            pos_weight=pos_weight_tensor
        )
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
        
        
        # backbone_reps = self.model.backbone(x.float())
        # logits = self.model.classifier(backbone_reps).squeeze(1)
        logits = self(x)
        preds = torch.sigmoid(logits.squeeze(1))
        
        pos_weight_tensor = torch.tensor([self.pos_weight], device=logits.device)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits.squeeze(1), 
            labels.float(), 
            pos_weight=pos_weight_tensor
        )
        
        # Store predictions and labels for later use
        self.val_preds.append(preds.detach().float())
        self.val_labels.append(labels.detach().float())
        
        if self.emit_predictions:
            for p, l, a in zip(preds, labels, accs):
                if l == 1 or p.item() > 0.5:
                    print(f"{a}\t{p.item() :.4f}\t{l.item() :.2f}")
        
        # Update all metrics
        self.accuracy(preds, labels)
        self.specificity(preds, labels)
        self.precision(preds, labels)
        self.recall(preds, labels)
        self.confusion_matrix(preds, labels)
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

        
        sensitivities = np.zeros(len(self.fpr_thresholds))
        specificities = np.zeros(len(self.sensitivity_thresholds))
        
        # Calculate metrics for all processes (needed for checkpointing)
        fpr, tpr, thresholds_roc = roc_curve(gathered_labels.cpu().numpy(), gathered_preds.cpu().numpy())
        roc_auc = auc(fpr, tpr)
        
        # Calculate additional metrics using optimal threshold (F1 maximizing)
        # We'll compute these after finding the best threshold

        # Compute sensitivity at each fpr threshold
        for i, fpr_threshold in enumerate(self.fpr_thresholds):
            idx = fpr <= fpr_threshold
            sensitivity = tpr[idx]
            if len(sensitivity) > 0:
                sensitivities[i] = max(sensitivity)
            else:
                sensitivities[i] = 0
        
        # Compute specificity at each sensitivity threshold
        for i, sens_threshold in enumerate(self.sensitivity_thresholds):
            idx = tpr >= sens_threshold
            specificity = 1 - fpr[idx]  # Convert FPR to specificity
            if len(specificity) > 0:
                specificities[i] = max(specificity)
            else:
                specificities[i] = 0
        
        # Create Precision-Recall curve and compute metrics
        precision_vals, recall_vals, thresholds_pr = precision_recall_curve(gathered_labels.cpu().numpy(), gathered_preds.cpu().numpy())
        avg_precision = average_precision_score(gathered_labels.cpu().numpy(), gathered_preds.cpu().numpy())
        
        # Find threshold that maximizes F1 score
        f1_scores = 2 * precision_vals * recall_vals / (precision_vals + recall_vals)
        max_f1_idx = np.argmax(f1_scores)
        threshold = thresholds_pr[max_f1_idx]
        best_recall = recall_vals[max_f1_idx]
        best_precision = precision_vals[max_f1_idx]
        best_f1 = f1_scores[max_f1_idx]
        
        # Calculate additional metrics using the SAME approach as original metrics
        # All metrics computed at the F1-maximizing point from the PR curve
        
        # PPV is the same as precision (already computed as best_precision)
        ppv = best_precision
        
        # For other metrics, we need to calculate them at the same F1-maximizing threshold
        # Apply the F1-maximizing threshold to get binary predictions
        binary_preds = (gathered_preds.cpu().numpy() >= threshold).astype(int)
        labels_np = gathered_labels.cpu().numpy().astype(int)
        
        # Calculate confusion matrix components at the F1-maximizing threshold
        tp = np.sum((binary_preds == 1) & (labels_np == 1))
        tn = np.sum((binary_preds == 0) & (labels_np == 0))
        fp = np.sum((binary_preds == 1) & (labels_np == 0))
        fn = np.sum((binary_preds == 0) & (labels_np == 1))
        
        # Calculate derived metrics at the same threshold
        specificity_val = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
        accuracy_f1_threshold = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        # Only create and log plots if we have CometML logger and are on main process
        if self.trainer.is_global_zero and isinstance(self.logger, CometLogger):
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

        # Log metrics with sync_dist=True so they are available for checkpointing across all processes
        self.log('recall', best_recall, sync_dist=True)
        self.log('fscore', best_f1, sync_dist=True)
        self.log('threshold', threshold, sync_dist=True)
        self.log('precision', best_precision, sync_dist=True)
        self.log('specificity', specificity_val, sync_dist=True)
        self.log('ppv', ppv, sync_dist=True)
        self.log('npv', npv, sync_dist=True)
        for sens, fpr_threshold in zip(sensitivities, self.fpr_thresholds):
            self.log(f'recall_at_fpr_{fpr_threshold}', sens, sync_dist=True)
        for spec, sens_threshold in zip(specificities, self.sensitivity_thresholds):
            self.log(f'specificity_at_recall_{sens_threshold}', spec, sync_dist=True)

        # Log the Area Under Precision-Recall Curve (AUPRC)
        # This is the same as average precision score
        self.log('auprc', avg_precision, sync_dist=True)

        # Process syncing is handled by lightning for these, and we want sync_dist=True to make sure things are synced across processes 
        self.log('accuracy', accuracy_f1_threshold, sync_dist=True)
        self.log('val_loss', self.validation_loss_mean.compute(), sync_dist=True)
        self.log('training_loss', self.training_loss_mean.compute(), sync_dist=True)
        self.log('learning_rate', lr)
    
        self.accuracy.reset()
        self.specificity.reset()
        self.precision.reset()
        self.recall.reset()
        self.confusion_matrix.reset()

    def configure_optimizers(self):
        # Create parameter groups for encoder/backbone vs other parameters
        encoder_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'encoder' in name.lower() or 'backbone' in name.lower():
                encoder_params.append(param)
            else:
                other_params.append(param)
        
        # Create optimizer with parameter groups
        param_groups = []
        if encoder_params:
            param_groups.append({'params': encoder_params, 'name': 'encoder_backbone'})
        if other_params:
            param_groups.append({'params': other_params, 'name': 'other'})
        
        # If no parameter groups were created (fallback), use all parameters
        if not param_groups:
            param_groups = [{'params': self.model.parameters(), 'name': 'all'}]
        
        optimizer = torch.optim.AdamW(param_groups, lr=0.001, weight_decay=0.001)
        
        # Choose scheduler based on freeze_encoder_iters
        if self.freeze_encoder_iters > 0:
            lrschedule = util.FreezeEncoderWarmupCosineLRScheduler(
                optimizer, self.max_lr, self.min_lr, self.warmup_iters, 
                self.lr_decay_iters, self.freeze_encoder_iters
            )
        else:
            lrschedule = util.WarmupCosineLRScheduler(
                optimizer, self.max_lr, self.min_lr, self.warmup_iters, self.lr_decay_iters
            )
        
        return [optimizer], [lrschedule]

    # Add these methods to specify checkpoint monitoring preferences
    @property
    def checkpoint_monitor(self):
        return self.checkpoint_monitor_metric
    
    @property
    def checkpoint_mode(self):
        return self.checkpoint_monitor_mode
        
    @property
    def comet_project(self):
        return self.comet_project_name


class ClassificationModel(pl.LightningModule):
    def __init__(self, model, num_classes, min_lr=0.00001, max_lr=0.0001, warmup_iters=20, lr_decay_iters=250, emit_predictions=False, ckpt_params=None, comet_project_name=None):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.comet_project_name = comet_project_name
        
        # Multi-class metrics
        from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
        self.accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.f1_score = MulticlassF1Score(num_classes=num_classes)
        self.confusion_matrix = ConfusionMatrix(task='multiclass', num_classes=num_classes)

        self.training_loss_mean = MeanMetric()
        self.validation_loss_mean = MeanMetric()
        self.emit_predictions = emit_predictions
        if ckpt_params is None:
            ckpt_params = {}
        ckpt_params['model_class'] = self.__class__.__name__
        ckpt_params['backbone_class'] = model.__class__.__name__
        if hasattr(model, 'model_conf'):
            ckpt_params['model_conf'] = model.model_conf
        else:
            ckpt_params['model_conf'] = {}
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
        loss = torch.nn.functional.cross_entropy(preds, labels.long(), label_smoothing=0.1)
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
        loss = torch.nn.functional.cross_entropy(preds, labels.long(), label_smoothing=0.1)
        
        # Get class predictions
        pred_classes = torch.argmax(preds, dim=1)
        
        # Store predictions and labels for later use
        self.val_preds.append(pred_classes.detach())
        self.val_labels.append(labels.detach())
        
        if self.emit_predictions:
            for p, l, a in zip(pred_classes, labels, accs):
                print(f"{a}\t{p.item()}\t{l.item()}")
        
        # Update metrics
        self.confusion_matrix(preds, labels.long())
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
        
        cm = self.confusion_matrix.compute()
        self.logger.experiment.log_confusion_matrix(matrix=cm.cpu().numpy(), step=self.current_epoch)

        # Log metrics
        self.log('accuracy', accuracy, sync_dist=True)
        self.log('f1_score', f1, sync_dist=True)
        self.log('val_loss', self.validation_loss_mean.compute(), sync_dist=True)
        self.log('training_loss', self.training_loss_mean.compute(), sync_dist=True)
        self.log('learning_rate', lr)
        
        # Reset metrics
        self.accuracy.reset()
        self.f1_score.reset()
        self.confusion_matrix.reset()

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
        return self.comet_project_name


class RegressionModel(pl.LightningModule):
    def __init__(self, model, min_lr=0.00001, max_lr=0.0001, warmup_iters=20, lr_decay_iters=250, emit_predictions=False, ckpt_params=None, comet_project_name=None):
        super().__init__()
        self.model = model
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.emit_predictions = emit_predictions
        self.comet_project_name = comet_project_name
        if ckpt_params is None:
            ckpt_params = {}
        ckpt_params['model_class'] = self.__class__.__name__
        ckpt_params['backbone_class'] = model.__class__.__name__
        if hasattr(model, 'model_conf'):
            ckpt_params['model_conf'] = model.model_conf
        else:
            ckpt_params['model_conf'] = {}
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
        logits = self(x)
        preds = F.sigmoid(logits.squeeze(1)) * 100.0
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
        logits = self(x)
        preds = F.sigmoid(logits.squeeze(1)) * 100.0
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
        return self.comet_project_name


class ContrastClassificationModel(pl.LightningModule):
    def __init__(
        self,
        model,                           # CombinedModel(btm, classifier backbone)
        emit_predictions: bool = False,
        ckpt_params: dict | None = None,
        min_lr: float = 0.00001,
        max_lr: float = 0.00025,
        num_classes: int = 1,
        warmup_iters: int = 10,
        lr_decay_iters: int = 80,
        freeze_encoder_iters: int = 0,
        checkpoint_monitor: str = "specificity_at_recall_0.99",
        checkpoint_mode: str = "max",
        comet_project_name: str | None = None,
        contrastive_weight: float = 0,
        #contrastive_warmup_steps: int=3000,
        #contrastive_ramp_steps: int=7000,
        report_key: str = "text_emb",
        proj_dim: int = 1024,
        init_temperature: float = 0.07,
        pos_weight: float | None = None,  
    ):
        super().__init__()

        self.num_classes = num_classes  # binary: 1 logit
        self.model = model   # CombinedModel (btm encoder)
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.freeze_encoder_iters = freeze_encoder_iters
        self.comet_project_name = comet_project_name
        self.contrastive_weight = contrastive_weight
        #self.contrastive_warmup_steps = contrastive_warmup_steps
        #self.contrastive_ramp_steps = contrastive_ramp_steps
        self.checkpoint_monitor_metric = checkpoint_monitor
        self.checkpoint_monitor_mode = checkpoint_mode
        self.report_key = report_key
        self.pos_weight = pos_weight
        self.current_contrastive_weight = 0.0
        self.contrastive_warmup_epochs= 2
        self.contrastive_ramp_epochs=3
        self.contrastive_end_epoch=10
        self.hard_neg_threshold = 0.0   # start here (p>0.5)
        self.hard_neg_weight = 2.0      # start 2x; later try 3–4 if FN stays stable
        

        # Contrastive loss
        self.InfoNCE_loss = InfoNCELoss()
        #self.supcon_loss = BinarySupervisedMultimodalContrastiveLoss(temperature=0.07)

        if ckpt_params is None:
            ckpt_params = {}

        # Read model hyperparameters
        output_scale_factor = ckpt_params.get("output_scale_factor", 1.0)

        # Get fused_dim from CombinedModel
        if not hasattr(self.model, "fused_dim"):
            raise ValueError(
                "CombinedModel must expose fused_dim, e.g. "
                "self.fused_dim = backbone.fused_dim inside CombinedModel.__init__"
            )
        self.fused_dim = self.model.fused_dim
        backbone_out_dim = self.fused_dim

        # Metrics
        self.accuracy = BinaryAccuracy()
        self.specificity = BinarySpecificity()
        self.precision = BinaryPrecision()   # PPV
        self.recall = BinaryRecall()
        self.confusion_matrix = ConfusionMatrix(task="binary")

        self.training_loss_mean = MeanMetric()
        self.validation_loss_mean = MeanMetric()
        self.emit_predictions = emit_predictions

        # Threshold sets for custom metrics
        self.fpr_thresholds = [0.01, 0.02, 0.05]
        self.sensitivity_thresholds = [0.95, 0.99, 0.995]

        from dinoflow.eval import ContrastClassificationHead  

        # Attach contrastive clf head on top of BTMTubes features
        self.head = ContrastClassificationHead(
            num_features=backbone_out_dim,
            num_classes=self.num_classes,
            proj_dim=proj_dim,
            output_scale_factor=output_scale_factor,
        )

        # Save hyperparams for checkpointing
        ckpt_params["model_class"] = self.__class__.__name__
        ckpt_params["backbone_class"] = model.__class__.__name__
        ckpt_params["model_conf"] = getattr(model, "model_conf", {})
        ckpt_params["contrastive_weight"] = contrastive_weight
        ckpt_params["proj_dim"] = proj_dim
        ckpt_params["init_temperature"] = init_temperature
        ckpt_params["backbone_out_dim"] = backbone_out_dim
        ckpt_params["pos_weight"] = pos_weight
        self.save_hyperparameters(ckpt_params)

        # Buffers for epoch-end metrics
        self.val_preds = []
        self.val_labels = []

    def update_contrastive_weight(self):
        epoch = self.current_epoch
        warm_epochs = self.contrastive_warmup_epochs
        ramp_epochs = self.contrastive_ramp_epochs
        end_epoch = self.contrastive_end_epoch
        target = self.contrastive_weight

        if epoch < warm_epochs:
            self.current_contrastive_weight = 0.0
        elif epoch < warm_epochs + ramp_epochs:
            t = (epoch - warm_epochs) / ramp_epochs
            self.current_contrastive_weight = t * target
        elif epoch < end_epoch:
            self.current_contrastive_weight = target
        else:
            self.current_contrastive_weight = 0.0


    # Forward
    def forward(self, batch):
        """
        batch: eventdict with 'b','t','m' → CombinedModel → features
        """
        feats = self.model(batch)              # (B, fused_dim)
        logits, z_flow = self.head(feats)      # logits: (B, 1), z_flow: (B, proj_dim)
        return logits, z_flow
    
    def on_train_epoch_start(self):
        self.update_contrastive_weight()

    def on_train_batch_start(self, batch, batch_idx):
        sch = self.lr_schedulers()
        if hasattr(sch, "set_iters"):
            sch.set_iters(self.global_step)


    # Training
 
    def training_step(self, batch, batch_idx):
        x, rowinfo = batch
        # labels: shape (B,)
        labels = rowinfo["label"].to(self.device)

        # text / report embeddings (already precomputed, proj_dim)
        z_rep = rowinfo[self.report_key].to(self.device).detach()  # (B, proj_dim)
        # Forward through combined model: get logits + flow projection
        logits, z_flow = self(x)          # logits: (B,1), z_flow: (B, proj_dim)

        # BCE labels: float, same shape as logits
       # labels_float = labels.float().view_as(logits)  # (B,1)

        #if self.pos_weight is not None:
            #pos_weight_tensor = torch.tensor([self.pos_weight], device=logits.device)
            #loss_cls = F.binary_cross_entropy_with_logits(
                #logits, labels_float, pos_weight=pos_weight_tensor
            #)
        #else:
            #loss_cls = F.binary_cross_entropy_with_logits(logits, labels_float)

        # Flatten to (B,)
        logits_1d = logits.view(-1)
        labels_1d = labels.view(-1).float()

        # Per-sample BCE (no reduction so we can reweight)
        base = F.binary_cross_entropy_with_logits(
            logits_1d, labels_1d, reduction="none"
        )

        weights = torch.ones_like(base)

        # Keep your existing pos_weight behavior (approx equivalent):
        if self.pos_weight is not None:
            # Multiply positive examples by pos_weight
            weights = weights * (1.0 + labels_1d * (float(self.pos_weight) - 1.0))

        # Hard-negative weighting: negatives that look positive
        #hard_neg_threshold = 0.0   # start here (p>0.5)
        #hard_neg_weight = 2.0      # start 2x; later try 3–4 if FN stays stable

        hard_neg = (labels_1d == 0) & (logits_1d > self.hard_neg_threshold)
        weights[hard_neg] *= self.hard_neg_weight

        self.log(
            "train/hard_neg_frac",
            hard_neg.float().mean(),
            on_step=True,
            prog_bar=False,
            sync_dist=True,
        )

        self.log(
            "train/hard_neg_count",
            hard_neg.sum(),
            on_step=True,
            prog_bar=False,
            sync_dist=True,
        )

        loss_cls = (base * weights).mean()


        # Supervised multimodal contrastive term (gated by *current* warmup weight)
        loss_con = logits.new_tensor(0.0)
        if self.current_contrastive_weight > 0.0:
            # labels for contrastive: int {0,1}
            #labels_int = labels.long()
            #oss_con = self.supcon_loss(z_flow, z_rep, labels_int)
            loss_con = self.InfoNCE_loss(z_flow, z_rep)

        loss = loss_cls + self.current_contrastive_weight * loss_con

        # Logging
        self.log("train_loss_cls", loss_cls.detach(), on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_loss_contrast", loss_con.detach(), on_step=True, on_epoch=True, sync_dist=True)
        self.log("contrastive_weight_now", self.current_contrastive_weight, on_step=True, sync_dist=True)

        self.training_loss_mean.update(loss.detach())

        return loss


    # Validation
    def on_validation_epoch_start(self):
        # Clear lists at epoch start
        self.val_preds = []
        self.val_labels = []

    def validation_step(self, batch, batch_idx):
        x, rowinfo = batch
        labels = rowinfo["label"].to(self.device)
        accs = rowinfo["ACCESSION"]
        z_rep = rowinfo[self.report_key].to(self.device).detach()

        logits, z_flow = self(x)             # logits: (B,1)
        labels = labels.float().view_as(logits)  # also (B,1)
        
        # BCE with optional pos_weight
        if self.pos_weight is not None:
            pos_weight_tensor = torch.tensor([self.pos_weight], device=logits.device)
            loss_cls = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight_tensor
            )
        else:
            loss_cls = F.binary_cross_entropy_with_logits(logits, labels)

        #loss_con = logits.new_tensor(0.0)
        
        if self.contrastive_weight > 0.0:
            loss_con = self.InfoNCE_loss(z_flow, z_rep)
            #labels_int = labels.long().view(-1)  # [B]
            #loss_con = self.supcon_loss(z_flow, z_rep, labels_int)

        #loss = loss_cls + self.contrastive_weight * loss_con
        loss = loss_cls  

        #self.log("val_loss_contrast", loss_con.detach(), on_step=True, on_epoch=True, sync_dist=True)
        self.log("val_loss_cls", loss_cls.detach(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # probs and labels as 1D tensors [B] for metrics & epoch-end curves
        probs = torch.sigmoid(logits).view(-1)      # shape [B]
        labels_flat = labels.view(-1)               # shape [B]

        # store for ROC/PR at epoch end
        self.val_preds.append(probs.detach())
        self.val_labels.append(labels_flat.detach())

        # optional hard preds for printing
        hard_preds = (probs >= 0.5).long()

        if self.emit_predictions:
            for p, l, a in zip(hard_preds, labels_flat.long(), accs):
                print(f"{a}\t{p.item()}\t{l.item()}")

        # torchmetrics: preds and targets must have SAME shape
        self.confusion_matrix(probs, labels_flat.long())
        self.accuracy(probs, labels_flat.long())
        self.validation_loss_mean.update(loss.detach())


    def on_validation_epoch_end(self):
        # LR for logging
        lrsched = self.lr_schedulers()
        lr = lrsched.get_last_lr()[0]

        accuracy_metric = self.accuracy.compute()

        # Gather predictions and labels from all processes
        world_size = getattr(self.trainer, "world_size", 1)
        if world_size and world_size > 1:
            gathered_preds = self.all_gather(torch.cat(self.val_preds)).flatten()
            gathered_labels = self.all_gather(torch.cat(self.val_labels)).flatten()
        else:
            gathered_preds = torch.cat(self.val_preds)
            gathered_labels = torch.cat(self.val_labels)

        gathered_preds = gathered_preds.float().detach().cpu()
        gathered_labels = gathered_labels.int().detach().cpu()

        y_scores = gathered_preds.numpy()
        y_true = gathered_labels.numpy()

        # Defaults
        best_f1 = float("NaN")
        best_precision = float("NaN")
        threshold = float("NaN")
        best_recall = float("NaN")
        sensitivities = np.zeros(len(self.fpr_thresholds))
        specificities = np.zeros(len(self.sensitivity_thresholds))

        # ROC
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Sensitivity at FPR thresholds
        for i, fpr_threshold in enumerate(self.fpr_thresholds):
            idx = fpr <= fpr_threshold
            sensitivity = tpr[idx]
            sensitivities[i] = max(sensitivity) if len(sensitivity) > 0 else 0.0

        # Specificity at sensitivity thresholds
        for i, sens_threshold in enumerate(self.sensitivity_thresholds):
            idx = tpr >= sens_threshold
            specificity = 1 - fpr[idx]
            specificities[i] = max(specificity) if len(specificity) > 0 else 0.0

        # PR curve
        precision_vals, recall_vals, thresholds_pr = precision_recall_curve(
            y_true, y_scores
        )
        avg_precision = average_precision_score(y_true, y_scores)

        # F1 over PR curve
        denom = precision_vals + recall_vals
        denom[denom == 0] = 1e-8
        f1_scores = 2 * precision_vals * recall_vals / denom
        max_f1_idx = np.argmax(f1_scores)

        if max_f1_idx < len(thresholds_pr):
            threshold = thresholds_pr[max_f1_idx]
        else:
            # edge case: when PR returns one extra point
            threshold = 0.5

        best_f1 = f1_scores[max_f1_idx]
        best_precision = precision_vals[max_f1_idx]
        best_recall = recall_vals[max_f1_idx]

        # Binary predictions at F1-max threshold
        binary_preds = (y_scores >= threshold).astype(int)

        tp = np.sum((binary_preds == 1) & (y_true == 1))
        tn = np.sum((binary_preds == 0) & (y_true == 0))
        fp = np.sum((binary_preds == 1) & (y_true == 0))
        fn = np.sum((binary_preds == 0) & (y_true == 1))

        specificity_val = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        accuracy_f1_threshold = (
            (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        )
        ppv = best_precision

        # Log ROC & PR figures to Comet on main process
        if self.trainer.is_global_zero and isinstance(self.logger, CometLogger):
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
            ax.plot([0, 1], [0, 1], "k--")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Receiver Operating Characteristic")
            ax.legend(loc="lower right")
            ax.grid(True)
            self.logger.experiment.log_figure(
                figure=fig, figure_name="ROC_Curve", step=self.current_epoch
            )
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(recall_vals, precision_vals, label=f"PR curve (AP = {avg_precision:.3f})")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall Curve")
            ax.legend(loc="lower left")
            ax.grid(True)
            self.logger.experiment.log_figure(
                figure=fig, figure_name="PR_Curve", step=self.current_epoch
            )
            plt.close(fig)

        # Log scalar metrics (these are what you'll likely checkpoint on)
        self.log("recall", best_recall, sync_dist=True)
        self.log("fscore", best_f1, sync_dist=True)
        self.log("threshold", threshold, sync_dist=True)
        self.log("precision", best_precision, sync_dist=True)
        self.log("specificity", specificity_val, sync_dist=True)
        self.log("ppv", ppv, sync_dist=True)
        self.log("npv", npv, sync_dist=True)

        for sens, fpr_threshold in zip(sensitivities, self.fpr_thresholds):
            self.log(f"recall_at_fpr_{fpr_threshold}", sens, sync_dist=True)
        for spec, sens_threshold in zip(specificities, self.sensitivity_thresholds):
            self.log(f"specificity_at_recall_{sens_threshold}", spec, sync_dist=True)

        self.log("auprc", avg_precision, sync_dist=True)
        self.log("accuracy", accuracy_f1_threshold, sync_dist=True)
        self.log("val_loss", self.validation_loss_mean.compute(), sync_dist=True)
        self.log("training_loss", self.training_loss_mean.compute(), sync_dist=True)
        self.log("learning_rate", lr)

        # Reset metrics for next epoch
        self.accuracy.reset()
        self.specificity.reset()
        self.precision.reset()
        self.recall.reset()
        self.confusion_matrix.reset()
        self.training_loss_mean.reset()
        self.validation_loss_mean.reset()

    # Optimizer / Scheduler
    def configure_optimizers(self):
        # Separate encoder/backbone vs other params (including head)
        encoder_params = []
        other_params = []

        # Use self.named_parameters so we include self.head as well
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "encoder" in name.lower() or "backbone" in name.lower():
                encoder_params.append(param)
            else:
                other_params.append(param)

        param_groups = []
        if encoder_params:
            param_groups.append({"params": encoder_params, "name": "encoder_backbone"})
        if other_params:
            param_groups.append({"params": other_params, "name": "other"})

        if not param_groups:
            param_groups = [{"params": self.parameters(), "name": "all"}]

        optimizer = torch.optim.AdamW(param_groups, lr=0.001, weight_decay=0.001)

        # Schedulers
        if self.freeze_encoder_iters > 0:
            lrschedule = util.FreezeEncoderWarmupCosineLRScheduler(
                optimizer,
                self.max_lr,
                self.min_lr,
                self.warmup_iters,
                self.lr_decay_iters,
                self.freeze_encoder_iters,
            )
        else:
            lrschedule = util.WarmupCosineLRScheduler(
                optimizer,
                self.max_lr,
                self.min_lr,
                self.warmup_iters,
                self.lr_decay_iters,
            )

        return [optimizer], [lrschedule]


    # Properties for checkpointing
    @property
    def checkpoint_monitor(self):
        return self.checkpoint_monitor_metric

    @property
    def checkpoint_mode(self):
        return self.checkpoint_monitor_mode

    @property
    def comet_project(self):
        return self.comet_project_name



import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

IGNORE_INDEX = -100


class MultiTaskClassificationModel(pl.LightningModule):
    """
    - Model outputs dict: task -> logits/preds
    - Dataset provides rowinfo["labels"][task] and rowinfo["label_masks"][task]
    - Primary task (binary) gets ROC/PR + thresholded metrics at epoch end
    - No gating, no contrast in-module; SWA is optional via Trainer (see eval._run_trainer + train.swa in YAML)
    """

    def __init__(
        self,
        model,
        task_defs: dict, # task -> {"type", "out_dim".}
        task_weights: dict,  # task -> float
        primary_task: str = "action_required",
        label_map: dict | None = None, # task -> key in rowinfo["labels"] (default identity)
        emit_predictions: bool = False,

        # opt/sched knobs 
        min_lr: float = 1e-5,
        max_lr: float = 2.5e-4,
        trunk_lr_mult: float = 0.3,
        warmup_epochs: int = 10,
        decay_end_epoch: int = 100,
        hold_epochs: int = 5,
        freeze_encoder_iters: int = 0,

        checkpoint_monitor: str = "val/mean_specificity_at_npv",
        checkpoint_mode: str = "max",

        # BCE pos_weight per task (optional)
        pos_weight: dict | None = None,

        # metrics
        fpr_thresholds=(0.01, 0.02, 0.05),
        sensitivity_thresholds=(0.95, 0.99, 0.995),

        # optimizer hyperparams
        weight_decay: float = 1e-3,
    ):
        super().__init__()
        self.model = model

        self.task_defs = task_defs
        self.task_weights = task_weights
        self.primary_task = primary_task
        self.label_map = label_map or {t: t for t in task_defs.keys()}
        self.pos_weight = pos_weight or {}

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.trunk_lr_mult = trunk_lr_mult
        self.warmup_epochs = warmup_epochs
        self.decay_end_epoch = decay_end_epoch
        self.hold_epochs = hold_epochs
        self.freeze_encoder_iters = freeze_encoder_iters
        self.weight_decay = weight_decay

        self.checkpoint_monitor_metric = checkpoint_monitor
        self.checkpoint_monitor_mode = checkpoint_mode

        self.emit_predictions = emit_predictions
        self.fpr_thresholds = list(fpr_thresholds)
        self.sensitivity_thresholds = list(sensitivity_thresholds)

        self.save_hyperparameters(ignore=["model"])

        # buffers for epoch-end metrics (primary task)
        self.val_preds = []
        self.val_labels = []
        self.val_masks = []

    # properties for checkpointing 
    @property
    def checkpoint_monitor(self):
        return self.checkpoint_monitor_metric

    @property
    def checkpoint_mode(self):
        return self.checkpoint_monitor_mode

    # forward
    def forward(self, x):
        out = self.model(x)
        if isinstance(out, tuple):
            out = out[0]
        return out

    # helpers for masking and gating for loss computation 
    def _get_y_m(self, rowinfo, task):
        """
        rowinfo["labels"][task], rowinfo["label_masks"][task]
        """
        key = self.label_map.get(task, task)

        if "labels" in rowinfo and isinstance(rowinfo["labels"], dict) and key in rowinfo["labels"]:
            y = rowinfo["labels"][key]
        else:
            y = rowinfo.get(key, None)

        if y is None:
            return None, None

        if "label_masks" in rowinfo and isinstance(rowinfo["label_masks"], dict) and key in rowinfo["label_masks"]:
            m = rowinfo["label_masks"][key].float()
        else:
            ttype = self.task_defs[task]["type"]
            if ttype in ("ce", "ordinal"):
                m = (y != IGNORE_INDEX).float()
            else:
                m = torch.isfinite(y).float()

        return y, m
    
    def _apply_parent_gate(self, rowinfo, task, parent_task="abnormal_pop"):
        #gates relevant tasks on parent task 
        y, m = self._get_y_m(rowinfo, task)
        if y is None:
            return None, None

        y_parent, m_parent = self._get_y_m(rowinfo, parent_task)
        if y_parent is None:
            return y, torch.zeros_like(m)

        gate = ((m_parent > 0) & (y_parent > 0.5)).float()
        m = m.float() * gate
        return y, m

    def _masked_bce(self, logits, y, m, pos_weight=None):
        logits = logits.view(-1)
        y = y.view(-1).float()
        m = m.view(-1).float()
        mask = m > 0
        if mask.sum() == 0:
            return None
        pw = torch.tensor([pos_weight], device=logits.device) if pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(logits[mask], y[mask], pos_weight=pw)

    def _masked_ce(self, logits, y, m):
        y = y.long()
        m = m.float()
        mask = (m > 0) & (y != IGNORE_INDEX)
        if mask.sum() == 0:
            return None
        return F.cross_entropy(logits[mask], y[mask])

    def _masked_reg(self, pred, y, m, kind="huber", delta=1.0):
        pred = pred.view(-1).float()
        y = y.view(-1).float()
        m = m.view(-1).float()
        mask = m > 0
        if mask.sum() == 0:
            return None
        if kind == "huber":
            return F.huber_loss(pred[mask], y[mask], delta=float(delta))
        if kind == "l1":
            return F.l1_loss(pred[mask], y[mask])
        return F.mse_loss(pred[mask], y[mask])

    #def _masked_ordinal(self, logits, y, m):
        #"""
        #CORAL style ordinal loss:
          #logits: [B, K-1]
          #y: [B] in 0..K-1 or IGNORE_INDEX
        #""
        #y = y.long()
        #m = m.float()
        #mask = (m > 0) & (y != IGNORE_INDEX)
        #if mask.sum() == 0:
            #return None

        #K_minus_1 = logits.shape[1]
        #th = torch.arange(K_minus_1, device=logits.device).view(1, -1)
        #y_clamped = torch.clamp(y, 0, K_minus_1)
        #y_exp = (y_clamped.view(-1, 1) > th).float()  # [B, K-1]

        #return F.binary_cross_entropy_with_logits(logits[mask], y_exp[mask])
    
    def _masked_ordinal(
        self,
        logits,
        y,
        m,
        *,
        threshold_weights=None,
        pos_weight_thresholds=None,
    ):
        """
        CORAL-style ordinal loss (K-1 cumulative binary problems).

        Args:
            logits: [B, K-1]
            y: [B] in 0..K-1 or IGNORE_INDEX
            threshold_weights: optional sequence of length K-1, non-negative.
                Weighted mean over thresholds: sum_i w_i * BCE_i / sum_i w_i.
                Use e.g. larger weights on high-burden cutpoints for disease burden.
            pos_weight_thresholds: optional sequence of length K-1, positive.
                BCE pos_weight per threshold (imbalance at each cumulative cut).
        """
        y = y.long()
        m = m.float()
        mask = (m > 0) & (y != IGNORE_INDEX)
        if mask.sum() == 0:
            return None

        logits = logits[mask]
        y = y[mask]

        K_minus_1 = logits.shape[1]
        th = torch.arange(K_minus_1, device=logits.device).view(1, -1)
        y = torch.clamp(y, 0, K_minus_1)
        y_exp = (y.view(-1, 1) > th).float()

        dev = logits.device
        dt = logits.dtype
        pw = None
        if pos_weight_thresholds is not None:
            pw = torch.as_tensor(pos_weight_thresholds, device=dev, dtype=dt).view(1, -1)
            if pw.numel() != K_minus_1:
                raise ValueError(
                    f"coral_pos_weight length {pw.numel()} != K-1={K_minus_1}"
                )
            if (pw <= 0).any():
                raise ValueError("coral_pos_weight values must be positive")

        bce = F.binary_cross_entropy_with_logits(
            logits, y_exp, pos_weight=pw, reduction="none"
        )
        per_thresh = bce.mean(dim=0)

        if threshold_weights is not None:
            tw = torch.as_tensor(threshold_weights, device=dev, dtype=dt)
            if tw.numel() != K_minus_1:
                raise ValueError(
                    f"coral_threshold_weights length {tw.numel()} != K-1={K_minus_1}"
                )
            if (tw < 0).any():
                raise ValueError("coral_threshold_weights must be non-negative")
            denom = tw.sum().clamp_min(1e-8)
            return (per_thresh * tw).sum() / denom
        return per_thresh.mean()

    def on_fit_start(self):
        total_steps = int(self.trainer.estimated_stepping_batches)
        max_epochs = int(self.trainer.max_epochs)
        steps_per_epoch = max(1, int(round(total_steps / max_epochs)))

        warmup_steps = int(self.warmup_epochs) * steps_per_epoch
        hold_steps = int(self.hold_epochs) * steps_per_epoch
        decay_end_steps = int(self.decay_end_epoch) * steps_per_epoch
        decay_end_steps = min(decay_end_steps, total_steps)

        sched = self.lr_schedulers()
        if isinstance(sched, (list, tuple)):
            sched = sched[0]
        if isinstance(sched, dict):
            sched = sched["scheduler"]

        sched.set_schedule(
            warmup_iters=warmup_steps,
            hold_iters=hold_steps,
            lr_decay_iters=decay_end_steps,
            decay_mode="end",
        )

        @rank_zero_only
        def _log():
            print(
                f"[sched] total_steps={total_steps} steps/epoch={steps_per_epoch} | "
                f"warmup_steps={warmup_steps} hold_steps={hold_steps} decay_end_steps={decay_end_steps}"
            )
        _log()
        
    def training_step(self, batch, batch_idx):
        x, rowinfo = batch
        out = self(x)

        total = torch.tensor(0.0, device=self.device)

        for task, cfg in self.task_defs.items():
            if task not in out:
                continue

            if task in {"reactive_vs_malignant", "acute_maturation", "clonality"}:
                y, m = self._apply_parent_gate(rowinfo, task, parent_task="abnormal_pop")
            else:
                y, m = self._get_y_m(rowinfo, task)

            if y is None:
                continue

            w = float(self.task_weights.get(task, 1.0))
            ttype = cfg["type"]

            if ttype == "bce":
                loss = self._masked_bce(out[task], y, m, pos_weight=self.pos_weight.get(task))
            elif ttype == "ce":
                loss = self._masked_ce(out[task], y, m)
            elif ttype == "reg":
                loss = self._masked_reg(
                    out[task], y, m,
                    kind=cfg.get("loss", "huber"),
                    delta=cfg.get("huber_delta", 1.0),
                )
            elif ttype == "ordinal":
                loss = self._masked_ordinal(
                    out[task],
                    y,
                    m,
                    threshold_weights=cfg.get("coral_threshold_weights"),
                    pos_weight_thresholds=cfg.get("coral_pos_weight"),
                )
            else:
                raise ValueError(f"Unknown task type: {ttype} for task={task}")

            if loss is None:
                continue

            self.log(f"train/loss_{task}_step",  loss.detach(), on_step=True,  on_epoch=False, sync_dist=False)
            self.log(f"train/loss_{task}_epoch", loss.detach(), on_step=False, on_epoch=True,  sync_dist=True)
            total = total + w * loss

        self.log("train/loss_total_step",  total.detach(), on_step=True,  on_epoch=False, prog_bar=False, sync_dist=False)
        self.log("train/loss_total_epoch", total.detach(), on_step=False, on_epoch=True,  prog_bar=True,  sync_dist=True)

        return total

    # validation
    #UPDATED: cache + F1 for all task types
    def on_validation_epoch_start(self):
        # primary-task epoch metrics caches
        self.val_preds = []
        self.val_labels = []
        self.val_masks = []

        # per-task epoch metrics cache (all tasks)
        # task -> dict with fields depending on type
        self.val_task_cache = {}


    def validation_step(self, batch, batch_idx):
        x, rowinfo = batch
        out = self(x)

        # Per-task validation losses
        total = torch.tensor(0.0, device=self.device)

        for task, cfg in self.task_defs.items():
            if task not in out:
                continue

            if task in {"reactive_vs_malignant", "acute_maturation", "clonality"}:
                y, m = self._apply_parent_gate(rowinfo, task, parent_task="abnormal_pop")
            else:
                y, m = self._get_y_m(rowinfo, task)

            if y is None:
                continue

            w = float(self.task_weights.get(task, 1.0))
            ttype = cfg["type"]

            # cache per-task outputs for epoch-end F1s
            if ttype == "bce":
                probs = torch.sigmoid(out[task].view(-1))
                c = self.val_task_cache.setdefault(task, {"type": "bce", "p": [], "y": [], "m": []})
                c["p"].append(probs.detach())
                c["y"].append(y.view(-1).float().detach())
                c["m"].append(m.view(-1).float().detach())

            elif ttype == "ce":
                logits = out[task]  # [B, C]
                c = self.val_task_cache.setdefault(task, {"type": "ce", "logits": [], "y": [], "m": []})
                c["logits"].append(logits.detach())
                c["y"].append(y.view(-1).long().detach())
                c["m"].append(m.view(-1).float().detach())

            elif ttype == "ordinal":
                logits = out[task]  # assumes [B, K] class logits for ordinal levels
                c = self.val_task_cache.setdefault(task, {"type": "ordinal", "logits": [], "y": [], "m": []})
                c["logits"].append(logits.detach())
                c["y"].append(y.view(-1).long().detach())
                c["m"].append(m.view(-1).float().detach())

            elif ttype == "reg":
                # "F1-like" only if you binarize regression at some cutoff (see epoch_end)
                pred = out[task].view(-1)
                c = self.val_task_cache.setdefault(task, {"type": "reg", "pred": [], "y": [], "m": []})
                c["pred"].append(pred.detach())
                c["y"].append(y.view(-1).float().detach())
                c["m"].append(m.view(-1).float().detach())

            # compute loss
            if ttype == "bce":
                loss = self._masked_bce(out[task], y, m, pos_weight=self.pos_weight.get(task))
            elif ttype == "ce":
                loss = self._masked_ce(out[task], y, m)
            elif ttype == "reg":
                loss = self._masked_reg(
                    out[task], y, m,
                    kind=cfg.get("loss", "huber"),
                    delta=cfg.get("huber_delta", 1.0),
                )
            elif ttype == "ordinal":
                loss = self._masked_ordinal(
                    out[task],
                    y,
                    m,
                    threshold_weights=cfg.get("coral_threshold_weights"),
                    pos_weight_thresholds=cfg.get("coral_pos_weight"),
                )
            else:
                raise ValueError(f"Unknown task type: {ttype} for task={task}")

            if loss is None:
                continue

            # per-task val loss logs
            self.log(f"val/loss_{task}_step",  loss.detach(), on_step=True,  on_epoch=False, sync_dist=False)
            self.log(f"val/loss_{task}_epoch", loss.detach(), on_step=False, on_epoch=True,  sync_dist=True)

            total = total + w * loss

        # weighted total val loss (mirrors train)
        self.log("val/loss_total_step",  total.detach(), on_step=True,  on_epoch=False, prog_bar=False, sync_dist=False)
        self.log("val/loss_total_epoch", total.detach(), on_step=False, on_epoch=True,  prog_bar=True,  sync_dist=True)

        # Primary-task cache for epoch-end metrics (ROC/PR/NPV checkpointing)
        pt = self.primary_task
        if pt not in out:
            return

        y, m = self._get_y_m(rowinfo, pt)
        if y is None:
            return

        logits = out[pt].view(-1)
        probs = torch.sigmoid(logits)

        y_flat = y.view(-1).float()
        m_flat = m.view(-1).float()

        self.val_preds.append(probs.detach())
        self.val_labels.append(y_flat.detach())
        self.val_masks.append(m_flat.detach())

        # keep your explicit primary loss names too (optional)
        if self.task_defs[pt]["type"] == "bce":
            loss_pt = self._masked_bce(out[pt], y, m, pos_weight=self.pos_weight.get(pt))
            if loss_pt is not None:
                self.log("val/loss_primary_step",  loss_pt.detach(), on_step=True,  on_epoch=False, prog_bar=False, sync_dist=False)
                self.log("val/loss_primary_epoch", loss_pt.detach(), on_step=False, on_epoch=True,  prog_bar=True,  sync_dist=True)


    def on_validation_epoch_end(self):
        import numpy as np
        from sklearn.metrics import (
            roc_curve, auc,
            precision_recall_curve,
            average_precision_score,
            f1_score,
        )

        world_size = getattr(self.trainer, "world_size", 1)


        # primary task (currently action_required) epoch-end metrics
        if len(self.val_preds) == 0:
            self.log("val/specificity_at_recall_0.99", 0.0, on_epoch=True, sync_dist=True)
            self.log("val/specificity_at_npv_0.95", 0.0, on_epoch=True, sync_dist=True)
            self.log("val/threshold_at_npv_0.95", 0.5, on_epoch=True, sync_dist=True)
            self.log("val/specificity_at_npv_0.90", 0.0, on_epoch=True, sync_dist=True)
            self.log("val/threshold_at_npv_0.90", 0.5, on_epoch=True, sync_dist=True)
            self.log("val/specificity_at_npv_0.97", 0.0, on_epoch=True, sync_dist=True)
            self.log("val/threshold_at_npv_0.97", 0.5, on_epoch=True, sync_dist=True)
            self.log("val/mean_specificity_at_npv", 0.0, on_epoch=True, sync_dist=True)
            self.log("val/auroc", 0.0, on_epoch=True, sync_dist=True)
            self.log("val/auprc", 0.0, on_epoch=True, sync_dist=True)
            self.log("val/accuracy", 0.0, on_epoch=True, sync_dist=True)

            # per-task F1 as zeros
            if hasattr(self, "val_task_cache"):
                for task in self.task_defs.keys():
                    safe_task = task.replace("/", "_")
                    self.log(f"val/{safe_task}_f1", 0.0, on_epoch=True, sync_dist=True)
            return

        preds  = torch.cat(self.val_preds)
        labels = torch.cat(self.val_labels)
        masks  = torch.cat(self.val_masks) if hasattr(self, "val_masks") else torch.ones_like(labels)

        if world_size and world_size > 1:
            preds  = self.all_gather(preds).flatten()
            labels = self.all_gather(labels).flatten()
            masks  = self.all_gather(masks).flatten()

        preds  = preds.float().detach().cpu()
        labels = labels.int().detach().cpu()
        masks  = masks.detach().cpu()

        use = masks > 0

        roc_auc = 0.0
        avg_precision = 0.0
        best_f1 = 0.0
        best_precision = 0.0
        best_recall = 0.0
        threshold_f1 = 0.5
        specificity_val = 0.0
        npv_f1 = 0.0
        accuracy_f1_threshold = 0.0

        specificity_at_npv_095 = 0.0
        threshold_at_npv_095   = 0.5

        specificity_at_npv_090 = 0.0
        threshold_at_npv_090   = 0.5

        specificity_at_npv_097 = 0.0
        threshold_at_npv_097   = 0.5

        mean_spec_at_npv = 0.0

        sensitivities = np.zeros(len(self.fpr_thresholds), dtype=np.float32)
        specificities = np.zeros(len(self.sensitivity_thresholds), dtype=np.float32)

        if int(use.sum()) > 0:
            y_scores = preds[use].numpy()
            y_true   = labels[use].numpy().astype(int)

            if np.unique(y_true).size >= 2:
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                roc_auc = float(auc(fpr, tpr))

                # sensitivity@FPR
                for i, fpr_thr in enumerate(self.fpr_thresholds):
                    idx = fpr <= fpr_thr
                    sensitivities[i] = float(np.max(tpr[idx])) if np.any(idx) else 0.0

                # specificity@recall
                for i, sens_thr in enumerate(self.sensitivity_thresholds):
                    idx = tpr >= sens_thr
                    specificities[i] = float(np.max(1.0 - fpr[idx])) if np.any(idx) else 0.0

                precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_true, y_scores)
                avg_precision = float(average_precision_score(y_true, y_scores))

                denom = precision_vals + recall_vals
                denom[denom == 0] = 1e-8
                f1_scores = 2 * precision_vals * recall_vals / denom
                max_f1_idx = int(np.argmax(f1_scores))
                threshold_f1 = float(thresholds_pr[max_f1_idx]) if max_f1_idx < len(thresholds_pr) else 0.5

                best_f1 = float(f1_scores[max_f1_idx])
                best_precision = float(precision_vals[max_f1_idx])
                best_recall = float(recall_vals[max_f1_idx])

                binary_preds = (y_scores >= threshold_f1).astype(int)
                tp = int(np.sum((binary_preds == 1) & (y_true == 1)))
                tn = int(np.sum((binary_preds == 0) & (y_true == 0)))
                fp = int(np.sum((binary_preds == 1) & (y_true == 0)))
                fn = int(np.sum((binary_preds == 0) & (y_true == 1)))

                specificity_val = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                npv_f1 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
                accuracy_f1_threshold = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

                # maximize specificity s.t. NPV >= target, for multiple targets
                thr = np.unique(y_scores)
                thr = np.concatenate(([thr.min() - 1e-6], thr, [thr.max() + 1e-6]))

                npv_results = {}

                for target_npv in [0.90, 0.95, 0.97]:
                    best_spec = -1.0
                    best_thr  = 0.5

                    for t in thr:
                        yhat = (y_scores >= t).astype(np.int32)

                        tn2 = int(np.sum((yhat == 0) & (y_true == 0)))
                        fp2 = int(np.sum((yhat == 1) & (y_true == 0)))
                        fn2 = int(np.sum((yhat == 0) & (y_true == 1)))

                        spec2 = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0.0
                        npv2  = tn2 / (tn2 + fn2) if (tn2 + fn2) > 0 else 0.0

                        if npv2 >= target_npv:
                            if (spec2 > best_spec) or (spec2 == best_spec and float(t) > best_thr):
                                best_spec = spec2
                                best_thr  = float(t)

                    if best_spec >= 0:
                        npv_results[target_npv] = (float(best_spec), float(best_thr))
                    else:
                        npv_results[target_npv] = (0.0, 0.5)

                specificity_at_npv_090, threshold_at_npv_090 = npv_results[0.90]
                specificity_at_npv_095, threshold_at_npv_095 = npv_results[0.95]
                specificity_at_npv_097, threshold_at_npv_097 = npv_results[0.97]

                mean_spec_at_npv = float(np.mean([
                    specificity_at_npv_090,
                    specificity_at_npv_095,
                    specificity_at_npv_097,
                ]))

                # plots (global zero only)
                if self.trainer.is_global_zero and isinstance(self.logger, CometLogger):
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
                    ax.plot([0, 1], [0, 1], "k--")
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    ax.set_title("Receiver Operating Characteristic")
                    ax.grid(True)
                    ax.legend(loc="lower right")
                    self.logger.experiment.log_figure(
                        figure=fig,
                        figure_name=f"ROC_Curve/{self.primary_task}",
                        step=self.global_step,
                    )
                    plt.close(fig)

                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.plot(recall_vals, precision_vals, label=f"PR (AP={avg_precision:.3f})")
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel("Recall")
                    ax.set_ylabel("Precision")
                    ax.set_title("Precision-Recall Curve")
                    ax.grid(True)
                    ax.legend(loc="lower left")
                    self.logger.experiment.log_figure(
                        figure=fig,
                        figure_name=f"PR_Curve/{self.primary_task}",
                        step=self.global_step,
                    )
                    plt.close(fig)

        # Per-task F1 for all tasks
        # - bce: F1@0.5
        # - ce: macro-F1
        # - ordinal: macro-F1 with CORAL decoding
        multiclass_avg = "macro"

        # Optional: provide per-reg-task cutoff for binarization (else we log 0.0)
        # e.g. self.reg_f1_cutoffs = {"some_reg_task": 0.5}
        reg_cutoffs = getattr(self, "reg_f1_cutoffs", {})
        world_size = getattr(self.trainer, "world_size", 1)

        if hasattr(self, "val_task_cache"):
            for task, cache in self.val_task_cache.items():
                safe_task = task.replace("/", "_")
                ttype = cache["type"]

                # gather mask first
                m = torch.cat(cache["m"])
                if world_size and world_size > 1:
                    m = self.all_gather(m).flatten()
                m = m.detach().float().cpu()  # float32 avoids bf16 numpy issues
                use_t = (m > 0).numpy()

                # coverage logs for all tasks
                #self.log(f"val/{safe_task}_n", float(use_t.sum()), on_epoch=True, sync_dist=True)
                #self.log(f"val/{safe_task}_frac_labeled", float(use_t.mean()), on_epoch=True, sync_dist=True)

                if use_t.sum() == 0:
                    self.log(f"val/{safe_task}_f1", 0.0, on_epoch=True, sync_dist=True)

                    if ttype == "ordinal":
                        self.log(f"val/{safe_task}_f1_weighted", 0.0, on_epoch=True, sync_dist=True)
                        self.log(f"val/{safe_task}_acc", 0.0, on_epoch=True, sync_dist=True)
                        #self.log(f"val/{safe_task}_within1", 0.0, on_epoch=True, sync_dist=True)
                        self.log(f"val/{safe_task}_mae", 0.0, on_epoch=True, sync_dist=True)
                    if ttype == "reg":
                        self.log(f"val/{safe_task}_mae", 0.0, on_epoch=True, sync_dist=True)
                        self.log(f"val/{safe_task}_mse", 0.0, on_epoch=True, sync_dist=True)
                        _cfg0 = self.task_defs.get(task, {})
                        if util.reg_reports_physical_metrics(_cfg0):
                            self.log(
                                f"val/{safe_task}_mae_physical",
                                0.0,
                                on_epoch=True,
                                sync_dist=True,
                            )
                            self.log(
                                f"val/{safe_task}_mse_physical",
                                0.0,
                                on_epoch=True,
                                sync_dist=True,
                            )
                    continue

                if ttype == "bce":
                    p = torch.cat(cache["p"])
                    y = torch.cat(cache["y"])
                    if world_size and world_size > 1:
                        p = self.all_gather(p).flatten()
                        y = self.all_gather(y).flatten()

                    # cast to float32 before .numpy() (bf16 -> numpy crash)
                    p = p.detach().float().cpu().numpy()
                    y = y.detach().float().cpu().numpy().astype(int)

                    y_true = y[use_t]
                    y_pred = (p >= 0.5).astype(int)[use_t]

                    f1 = f1_score(y_true, y_pred, zero_division=0)

                elif ttype == "ce":
                    logits = torch.cat(cache["logits"], dim=0)
                    y = torch.cat(cache["y"])
                    if world_size and world_size > 1:
                        logits = self.all_gather(logits)
                        y = self.all_gather(y).flatten()
                        if logits.dim() == 3:
                            logits = logits.reshape(-1, logits.shape[-1])

                    logits = logits.detach().float().cpu()
                    y = y.detach().cpu().numpy().astype(int)

                    ypred = torch.argmax(logits, dim=-1).cpu().numpy().astype(int)

                    y_true = y[use_t]
                    y_pred = ypred[use_t]

                    f1 = f1_score(y_true, y_pred, average=multiclass_avg, zero_division=0)

                    # per-class support for CE
                    classes_present, counts = np.unique(y_true, return_counts=True)
                    for cls, cnt in zip(classes_present, counts):
                        self.log(
                            f"val/{safe_task}_support_class_{int(cls)}",
                            float(cnt),
                            on_epoch=True,
                            sync_dist=True
                        )

                elif ttype == "ordinal":
                    logits = torch.cat(cache["logits"], dim=0)
                    y = torch.cat(cache["y"])
                    if world_size and world_size > 1:
                        logits = self.all_gather(logits)
                        y = self.all_gather(y).flatten()
                        if logits.dim() == 3:
                            logits = logits.reshape(-1, logits.shape[-1])

                    logits = logits.detach().float().cpu()
                    y = y.detach().cpu().numpy().astype(int)

                    # CORAL decoding: number of passed thresholds = predicted class
                    probs = torch.sigmoid(logits)
                    ypred = (probs >= 0.5).sum(dim=-1).cpu().numpy().astype(int)

                    y_true = y[use_t]
                    y_pred = ypred[use_t]

                    f1 = f1_score(y_true, y_pred, average=multiclass_avg, zero_division=0)
                    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
                    acc = float((y_true == y_pred).mean())
                    within1 = float((np.abs(y_true - y_pred) <= 1).mean())
                    mae = float(np.abs(y_true - y_pred).mean())

                    self.log(f"val/{safe_task}_f1_weighted", float(f1_weighted), on_epoch=True, sync_dist=True)
                    self.log(f"val/{safe_task}_acc", float(acc), on_epoch=True, sync_dist=True)
                    self.log(f"val/{safe_task}_within1", float(within1), on_epoch=True, sync_dist=True)
                    self.log(f"val/{safe_task}_mae", float(mae), on_epoch=True, sync_dist=True)

                    # per-class support for ordinal
                    classes_present, counts = np.unique(y_true, return_counts=True)
                    #for cls, cnt in zip(classes_present, counts):
                    #    self.log(
                    #        f"val/{safe_task}_support_class_{int(cls)}",
                    #        float(cnt),
                    #        on_epoch=True,
                    #        sync_dist=True
                    #    )

                elif ttype == "reg":
                    pred = torch.cat(cache["pred"])
                    y = torch.cat(cache["y"])
                    if world_size and world_size > 1:
                        pred = self.all_gather(pred).flatten()
                        y = self.all_gather(y).flatten()

                    pred = pred.detach().float().cpu().numpy()  # bf16-safe
                    y = y.detach().float().cpu().numpy()

                    cfg_task = self.task_defs.get(task, {})
                    y_true = y[use_t]
                    y_pred = pred[use_t]
                    mae = float(np.abs(y_true - y_pred).mean())
                    mse = float(((y_true - y_pred) ** 2).mean())
                    self.log(f"val/{safe_task}_mae", mae, on_epoch=True, sync_dist=True)
                    self.log(f"val/{safe_task}_mse", mse, on_epoch=True, sync_dist=True)

                    if util.reg_reports_physical_metrics(cfg_task):
                        y_phys = util.reg_training_to_physical(y_true, cfg_task)
                        pred_phys = util.reg_training_to_physical(y_pred, cfg_task)
                        mae_p = float(np.abs(y_phys - pred_phys).mean())
                        mse_p = float(((y_phys - pred_phys) ** 2).mean())
                        self.log(
                            f"val/{safe_task}_mae_physical",
                            mae_p,
                            on_epoch=True,
                            sync_dist=True,
                        )
                        self.log(
                            f"val/{safe_task}_mse_physical",
                            mse_p,
                            on_epoch=True,
                            sync_dist=True,
                        )

                    thr = reg_cutoffs.get(task, None)
                    if thr is None:
                        f1 = 0.0
                    else:
                        thr = float(thr)
                        y_phys_full = util.reg_training_to_physical(y, cfg_task)
                        pred_phys_full = util.reg_training_to_physical(pred, cfg_task)
                        y_true_bin = (y_phys_full >= thr).astype(int)[use_t]
                        y_pred_bin = (pred_phys_full >= thr).astype(int)[use_t]
                        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

                    if (
                        util.reg_reports_physical_metrics(cfg_task)
                        and self.trainer.is_global_zero
                        and isinstance(self.logger, CometLogger)
                    ):
                        plot_range = getattr(self, "reg_plot_range", {}).get(
                            task, (0.0, 100.0)
                        )
                        low, high = float(plot_range[0]), float(plot_range[1])
                        y_plot = util.reg_training_to_physical(y_true, cfg_task)
                        pred_plot = np.clip(
                            util.reg_training_to_physical(y_pred, cfg_task), low, high
                        )
                        fig, ax = plt.subplots(figsize=(10, 8))
                        ax.scatter(y_plot, pred_plot, alpha=0.5)
                        ax.plot([low, high], [low, high], "r--", label="y=x")
                        ax.set_xlim(low, high)
                        ax.set_ylim(low, high)
                        ax.set_xlabel("Actual (physical units)")
                        ax.set_ylabel("Predicted (physical, clamped)")
                        ax.set_title(f"{safe_task} (physical scale, MAE={mae_p:.3f})")
                        ax.legend()
                        ax.grid(True)
                        self.logger.experiment.log_figure(
                            figure=fig,
                            figure_name=f"Predictions_vs_Actual/{safe_task}",
                            step=self.global_step,
                        )
                        plt.close(fig)

                else:
                    f1 = 0.0

                self.log(f"val/{safe_task}_f1", float(f1), on_epoch=True, sync_dist=True)

        # scalar logs: stable names (primary task)
        self.log("val/recall", best_recall, on_epoch=True, sync_dist=True)
        self.log("val/fscore", best_f1, on_epoch=True, sync_dist=True)
        self.log("val/threshold", threshold_f1, on_epoch=True, sync_dist=True)
        self.log("val/precision", best_precision, on_epoch=True, sync_dist=True)
        self.log("val/specificity", float(specificity_val), on_epoch=True, sync_dist=True)
        self.log("val/ppv", best_precision, on_epoch=True, sync_dist=True)
        self.log("val/npv", float(npv_f1), on_epoch=True, sync_dist=True)

        for sens, fpr_thr in zip(sensitivities, self.fpr_thresholds):
            self.log(f"val/recall_at_fpr_{fpr_thr:.3f}", float(sens), on_epoch=True, sync_dist=True)

        for spec, sens_thr in zip(specificities, self.sensitivity_thresholds):
            self.log(f"val/specificity_at_recall_{sens_thr:.2f}", float(spec), on_epoch=True, sync_dist=True)

        if 0.99 in [float(x) for x in self.sensitivity_thresholds]:
            idx = int(np.argmin(np.abs(np.array(self.sensitivity_thresholds, dtype=float) - 0.99)))
            self.log("val/specificity_at_recall_0.99", float(specificities[idx]), on_epoch=True, sync_dist=True)
        else:
            self.log("val/specificity_at_recall_0.99", 0.0, on_epoch=True, sync_dist=True)

        self.log("val/specificity_at_npv_0.95", float(specificity_at_npv_095), on_epoch=True, sync_dist=True)
        self.log("val/threshold_at_npv_0.95",  float(threshold_at_npv_095),   on_epoch=True, sync_dist=True)

        self.log("val/specificity_at_npv_0.90", float(specificity_at_npv_090), on_epoch=True, sync_dist=True)
        self.log("val/threshold_at_npv_0.90",  float(threshold_at_npv_090),   on_epoch=True, sync_dist=True)

        self.log("val/specificity_at_npv_0.97", float(specificity_at_npv_097), on_epoch=True, sync_dist=True)
        self.log("val/threshold_at_npv_0.97",  float(threshold_at_npv_097),   on_epoch=True, sync_dist=True)

        self.log("val/mean_specificity_at_npv", float(mean_spec_at_npv), on_epoch=True, sync_dist=True)

        self.log("val/auroc", roc_auc, on_epoch=True, sync_dist=True)
        self.log("val/auprc", avg_precision, on_epoch=True, sync_dist=True)
        self.log("val/accuracy", float(accuracy_f1_threshold), on_epoch=True, sync_dist=True)

        # clear buffers every epoch
        self.val_preds.clear()
        self.val_labels.clear()
        if hasattr(self, "val_masks"):
            self.val_masks.clear()
        if hasattr(self, "val_task_cache"):
            self.val_task_cache.clear()


    def configure_optimizers(self):
        encoder_params, other_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "encoder" in name.lower() or "backbone" in name.lower():
                encoder_params.append(p)
            else:
                other_params.append(p)

        trunk_mult = float(getattr(self, "trunk_lr_mult", 0.3))

        max_lr_heads = float(self.max_lr)
        min_lr_heads = float(self.min_lr)
        max_lr_trunk = max_lr_heads * trunk_mult
        min_lr_trunk = min_lr_heads * trunk_mult

        param_groups = []
        max_lrs, min_lrs = [], []

        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": max_lr_trunk, "name": "trunk"})
            max_lrs.append(max_lr_trunk); min_lrs.append(min_lr_trunk)

        if other_params:
            param_groups.append({"params": other_params, "lr": max_lr_heads, "name": "heads"})
            max_lrs.append(max_lr_heads); min_lrs.append(min_lr_heads)

        optimizer = torch.optim.AdamW(param_groups, weight_decay=float(self.weight_decay))

        # Use safe defaults; on_fit_start will set the real schedule in steps.
        sched = util.WarmupHoldCosineLRScheduler(
            optimizer,
            max_lrs=max_lrs,
            min_lrs=min_lrs,
            warmup_iters=0,
            hold_iters=0,
            lr_decay_iters=1,
            decay_mode="end",
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
                "frequency": 1,
            },
        }