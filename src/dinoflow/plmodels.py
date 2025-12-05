import torch
import pytorch_lightning as pl
from torchmetrics import MeanMetric, MeanSquaredError, MeanAbsoluteError, R2Score, ConfusionMatrix
from torchmetrics.classification import BinaryAccuracy, BinarySpecificity, BinaryPrecision, BinaryRecall
from torchmetrics.aggregation import MeanMetric, SumMetric
import torch.nn.functional as F

from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import CometLogger
from dinoflow import util
from dinoflow.loss import InfoNCELoss

class BinaryClassificationModel(pl.LightningModule):
    def __init__(self, model, min_lr=0.00001, max_lr=0.00025, warmup_iters=10, lr_decay_iters=80, emit_predictions=False, ckpt_params=None, num_classes=1, comet_project_name=None, freeze_encoder_iters=0):
        super().__init__()
        assert num_classes==1, "Only one class permitted for binary"
        self.model = model #
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.freeze_encoder_iters = freeze_encoder_iters
        self.accuracy = BinaryAccuracy()
        self.training_loss_mean = MeanMetric()
        self.validation_loss_mean = MeanMetric()
        self.emit_predictions = emit_predictions
        self.comet_project_name = comet_project_name
        # These are the thresholds at which we compute vthe sensitivity
        # Probably best not to change them
        self.fpr_thresholds = [0.01, 0.02, 0.05]
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
        
        
        # backbone_reps = self.model.backbone(x.float())
        # logits = self.model.classifier(backbone_reps).squeeze(1)
        logits = self(x)
        preds = torch.sigmoid(logits.squeeze(1))
        
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(1), labels.float())
        
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

        
        sensitivities = np.zeros(len(self.fpr_thresholds))
        # Only create and log the plot on the main process
        if self.trainer.is_global_zero and isinstance(self.logger, CometLogger):
           
            fpr, tpr, thresholds = roc_curve(gathered_labels.cpu().numpy(), gathered_preds.cpu().numpy())
            roc_auc = auc(fpr, tpr)

            # Compute sensitivity at each fpr threshold
            for i, fpr_threshold in enumerate(self.fpr_thresholds):
                idx = fpr <= fpr_threshold
                sensitivity = tpr[idx]
                if len(sensitivity) > 0:
                    sensitivities[i] = max(sensitivity)
                else:
                    sensitivities[i] = 0
            
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
        for sens, fpr_threshold in zip(sensitivities, self.fpr_thresholds):
            self.log(f'recall_at_fpr_{fpr_threshold}', sens)

        # Log the Area Under Precision-Recall Curve (AUPRC)
        # This is the same as average precision score
        if self.trainer.is_global_zero and isinstance(self.logger, CometLogger):
            self.log('auprc', avg_precision)
        else:
            self.log('auprc', float("NaN"))

        # Process syncing is handled by lightning for these, and we want sync_dist=True to make sure things are synced across processes 
        self.log('accuracy', accuracy, sync_dist=True)
        self.log('val_loss', self.validation_loss_mean.compute(), sync_dist=True)
        self.log('training_loss', self.training_loss_mean.compute(), sync_dist=True)
        self.log('learning_rate', lr)
    
        self.accuracy.reset()

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
        return 'val_loss'
    
    @property
    def checkpoint_mode(self):
        return 'min'
        
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
        model,                           # CombinedModel(btm, classifier)
        emit_predictions=False,
        ckpt_params=None,
        min_lr=1e-5,
        max_lr=1e-4,
        num_classes = 1,
        warmup_iters=20,
        lr_decay_iters=250,
        freeze_encoder_iters=0,
        checkpoint_monitor='val_loss', 
        checkpoint_mode='min',
        comet_project_name=None,
        contrastive_weight: float = 0.1,
        report_key: str = "text_emb",
        proj_dim: int = 384,
        init_temperature: float = 0.07,
    ):
        super().__init__()
        self.num_classes = num_classes #"Only one class permitted for binary"
        # Store the combined model (backbone + head)
        self.model = model   # CombinedModel
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.freeze_encoder_iters = freeze_encoder_iters
        self.comet_project_name = comet_project_name
        self.contrastive_weight = contrastive_weight
        self.checkpoint_monitor_metric = checkpoint_monitor
        self.checkpoint_monitor_mode = checkpoint_mode
        self.report_key = report_key
        self.InfoNCE_loss = InfoNCELoss()

        if ckpt_params is None:
            ckpt_params = {}

        # Read model hyperparameters
        #self.num_classes = ckpt_params.get("output_classes", 2)
        output_scale_factor = ckpt_params.get("output_scale_factor", 1.0)

        # Get fused_dim from CombinedModel, NOT btm
        if not hasattr(self.model, "fused_dim"):
            raise ValueError(
                "CombinedModel must expose fused_dim, e.g., "
                "self.fused_dim = backbone.fused_dim inside CombinedModel.__init__"
            )

        self.fused_dim = self.model.fused_dim
        backbone_out_dim = self.fused_dim
        
        # metrics
        self.accuracy = BinaryAccuracy()
        self.specificity = BinarySpecificity()
        self.precision = BinaryPrecision()  # This is PPV
        self.recall = BinaryRecall()        # For NPV calculation
        self.confusion_matrix = ConfusionMatrix(task='binary')

        self.training_loss_mean = MeanMetric()
        self.validation_loss_mean = MeanMetric()
        self.emit_predictions = emit_predictions

        # These are the thresholds at which we compute the sensitivity
        # Probably best not to change them
        self.fpr_thresholds = [0.01, 0.02, 0.05]
        # Sensitivity thresholds for computing specificity (equivalent to FNR thresholds)
        # FNR = 1 - Sensitivity, so sensitivity 0.99 = FNR 0.01
        self.sensitivity_thresholds = [0.95, 0.99, 0.995]

        from dinoflow.eval import ContrastClassificationHead
        # attach contrastive clf head on top of BTMTubes features
        self.head = ContrastClassificationHead(
            num_features=backbone_out_dim,
            num_classes=self.num_classes,
            proj_dim=proj_dim,
            output_scale_factor=output_scale_factor,
        )

        ckpt_params['model_class'] = self.__class__.__name__
        ckpt_params['backbone_class'] = model.__class__.__name__
        ckpt_params['model_conf'] = getattr(model, 'model_conf', {})
        ckpt_params['contrastive_weight'] = contrastive_weight
        ckpt_params['proj_dim'] = proj_dim
        ckpt_params['init_temperature'] = init_temperature
        ckpt_params['backbone_out_dim'] = backbone_out_dim
        self.save_hyperparameters(ckpt_params)

        self.val_preds = []
        self.val_labels = []

    def forward(self, batch):
        # batch: eventdict with 'b','t','m'
        feats = self.model(batch) # (B, 3 * model_dim)
        logits, z_flow = self.head(feats) # head adds clf + projection
        return logits, z_flow

    def training_step(self, batch, batch_idx):
        x, rowinfo = batch
        labels = rowinfo['label'].to(self.device)
        z_rep = rowinfo[self.report_key].to(self.device).detach()   # (B, proj_dim)

        logits, z_flow = self(x)

        labels=labels.float().view_as(logits)
        # classification on logits from head
        loss_cls = F.binary_cross_entropy_with_logits(logits, labels)

        # contrastive term (with infonce loss)
        loss_con = self.InfoNCE_loss(z_flow, z_rep)
        loss = loss_cls + self.contrastive_weight * loss_con

        self.log("train_loss_cls", loss_cls.detach(), on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_loss_contrast", loss_con.detach(), on_step=True, on_epoch=True, sync_dist=True)
        self.training_loss_mean.update(loss.detach())
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, rowinfo = batch
        labels = rowinfo['label'].to(self.device)
        accs = rowinfo['ACCESSION']
        z_rep = rowinfo[self.report_key].to(self.device).detach()

        logits, z_flow = self(x)

        labels=labels.float().view_as(logits)

        # classification on logits from head
        loss_cls = F.binary_cross_entropy_with_logits(logits, labels)

    
        loss_con = self.InfoNCE_loss(z_flow, z_rep)
        loss = loss_cls + self.contrastive_weight * loss_con
        self.log("val_loss_contrast", loss_con.detach(), on_step=True, on_epoch=True, sync_dist=True)
        self.log("val_loss_cls", loss_cls.detach(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        pred_classes = torch.argmax(logits, dim=1)

        self.val_preds.append(pred_classes.detach())
        self.val_labels.append(labels.detach())

        if self.emit_predictions:
            for p, l, a in zip(pred_classes, labels, accs):
                print(f"{a}\t{p.item()}\t{l.item()}")

        self.confusion_matrix(logits, labels.long())
        self.accuracy(logits, labels.long())
        #self.f1_score(logits, labels.long())
        self.validation_loss_mean.update(loss.detach())


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
