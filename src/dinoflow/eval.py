
import logging

import typer

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryPrecisionRecallCurve, BinaryF1Score, BinaryRecall, BinaryPrecision, BinaryAccuracy

from torch.utils.data import DataLoader
import numpy as np


from dinoflow.models import TubeEncoderWithProjection
from dinoflow.data import TubeData, collate_fn
from dinoflow import util
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CometLogger

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


class ClassificationModel(pl.LightningModule):
    def __init__(self, backbone, classifier, min_lr=0.00001, max_lr=0.001, warmup_iters=100, lr_decay_iters=5000):
        super().__init__()
        self.model = CombinedModel(backbone, classifier)
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.accuracy = BinaryAccuracy()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.f1score = BinaryF1Score()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat.squeeze(1), y.float())
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, labels = batch
        preds = self(x).squeeze(-1)
        self.accuracy(preds, labels > 0.5)
        self.precision(preds, labels > 0.5)
        self.recall(preds, labels > 0.5)
        self.f1score(preds, labels > 0.5)

    def on_validation_epoch_end(self):
        lrsched = self.lr_schedulers()
        lr = lrsched.get_last_lr()[0]
        accuracy = self.accuracy.compute()
        precision = self.precision.compute()
        recall = self.recall.compute()
        fscore = self.f1score.compute()

        self.log('precision', precision)
        self.log('accuracy', accuracy)
        self.log('recall', recall)
        self.log('fscore', fscore)
        self.log('learning_rate', lr)
    
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1score.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        lrschedule = util.WarmupCosineLRScheduler(optimizer, self.max_lr, self.min_lr, self.warmup_iters, self.lr_decay_iters)
        return [optimizer], [lrschedule]


def load_checkpoint(path):
    """
    Load a checkpoint from a file
    """
    ckpt = torch.load(path, weights_only=False, map_location=DEVICE)
    modelconf = ckpt['modelconf']    
    teacher = TubeEncoderWithProjection(num_features=modelconf['num_features'], model_embed_dim=modelconf['model_dim'], layers=modelconf['layers'], heads=modelconf['heads'], hidden_dim=modelconf['hidden_dim'], projection_dim=modelconf['projection_dim']).to(DEVICE)

    teacher.load_state_dict(ckpt['teacher'])
    return teacher.tube_encoder


@app.command()
def train(run_name, train_labels, test_labels, backbone: str, checkpoint: str = None, freeze_backbone: bool = False, batch_size: int=16, events: int = 4096, epochs: int = 25) :
    """
    Evaluate the model on the test set
    """
    logger.info(f"Loading backbone from {backbone}")
    backbone = load_checkpoint(backbone)
    classifier = ClassificationHead(backbone.cls_token.shape[-1], 1)
    model = ClassificationModel(backbone, classifier)

    if checkpoint is not None:
        logger.info(f"Loading full model checkpoint from {checkpoint}")
        model = ClassificationModel.load_from_checkpoint(checkpoint, backbone=backbone, classifier=classifier)    
    
    if freeze_backbone:
        logger.info("Freezing backbone")
        backbone.eval() # freeze backbone
        for p in backbone.parameters():
            p.requires_grad = False
    else:
        logger.info("Unfreezing backbone")
        backbone.train()

    torch.set_float32_matmul_precision('medium')
    
    traindata = TubeData(train_labels, tubes_to_return=["m"], events_to_return=int(events))
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=16)
    logger.info(f"Loaded {len(trainloader.dataset)} samples for training")

    valdata = TubeData(test_labels, tubes_to_return=["m"], events_to_return=int(events))
    valloader = DataLoader(valdata, batch_size=batch_size, shuffle=False, num_workers=16)
    logger.info(f"Loaded {len(valloader.dataset)} samples for val")

    comet_logger = CometLogger(
            workspace="brendan",  # Optional
            save_dir="dinoflow_classifier_runs",  # Optional
            project_name="dinoflow-classifier",  # Optional
            experiment_name=run_name,  # Optional
        )

    trainer = pl.Trainer(max_epochs=epochs,
                        accelerator='auto',
                        precision="bf16-mixed",
                        callbacks=[
                            ModelCheckpoint(monitor='fscore', mode='max', save_top_k=1, save_last=True),
                            LearningRateMonitor(logging_interval='step'),
                        ],
                        logger=comet_logger)

    trainer.fit(model, trainloader, valloader)


if __name__ == "__main__":
    app()
