import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import BinaryAccuracy, MeanMetric, MeanSquaredError, MeanAbsoluteError, R2Score
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
import numpy as np
import matplotlib.pyplot as plt
from comet_ml import CometLogger
from dinoflow.util import WarmupCosineLRScheduler


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
        if ckpt_params is None:
            ckpt_params = {}
        ckpt_params['model_class'] = self.__class__.__name__
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
        if ckpt_params is None:
            ckpt_params = {}
        ckpt_params['model_class'] = self.__class__.__name__
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
        if ckpt_params is None:
            ckpt_params = {}
        ckpt_params['model_class'] = self.__class__.__name__
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
