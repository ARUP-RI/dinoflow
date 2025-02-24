
import logging

import typer

import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import DataLoader
import numpy as np

from sklearn.metrics import precision_recall_fscore_support

from dinoflow.models import TubeEncoderWithProjection
from dinoflow.data import TubeData, collate_fn
from dinoflow import util
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, CometLogger

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
            nn.Sigmoid()
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
    def __init__(self, backbone, classifier, min_lr=0.00001, max_lr=0.001, warmup_iters=500, lr_decay_iters=1000):
        super().__init__()
        self.model = CombinedModel(backbone, classifier)
        self.all_val_predictions = []
        self.all_val_labels = []
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.BCELoss()(y_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.all_val_predictions.append(y_hat)
        self.all_val_labels.append(y)
    

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.all_val_predictions)
        all_labels = torch.cat(self.all_val_labels)
        threshold = find_best_threshold(all_preds, all_labels)
        precision, recall, fscore, support = precision_recall_fscore_support(all_labels, all_preds > threshold, average='binary')
        self.log('precision', precision)
        self.log('recall', recall)
        self.log('fscore', fscore)
        self.log('threshold', threshold)
        self.all_val_predictions = []
        self.all_val_labels = []
    
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


def find_best_threshold(predictions, labels):
    best_fscore = 0
    best_threshold = 0
    for threshold in np.arange(0.00001, 0.9999, 0.1):
        precision, recall, fscore, support = precision_recall_fscore_support(labels, predictions > threshold, average='binary')
        if fscore > best_fscore:
            best_fscore = fscore
            best_threshold = threshold
    return best_threshold


@app.command()
def train(run_name, train_labels, test_labels, checkpoint, freeze_backbone=False, batch_size: int=16, events: int = 4096, epochs: int = 25) :
    """
    Evaluate the model on the test set
    """
    backbone = load_checkpoint(checkpoint).to(DEVICE)
    if freeze_backbone:
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
    else:
        backbone.train()

    classifier = ClassificationHead(backbone.cls_token.shape[-1], 1).to(DEVICE)
    model = CombinedModel(backbone, classifier)

    traindata = TubeData(train_labels, tubes_to_return=["m"], events_to_return=int(events))
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True)
    logger.info(f"Loaded {len(trainloader.dataset)} samples for training")

    valdata = TubeData(test_labels, tubes_to_return=["m"], events_to_return=int(events))
    valloader = DataLoader(valdata, batch_size=batch_size, shuffle=False)
    logger.info(f"Loaded {len(valloader.dataset)} samples for val")

    trainer = pl.Trainer(max_epochs=epochs,
                        accelerator='auto',
                        devices=1,
                        callbacks=[
                            ModelCheckpoint(monitor='fscore', mode='max', save_top_k=1, save_last=True),
                            LearningRateMonitor(logging_interval='step'),
                            CometLogger(project_name="dinoflow-classifier", experiment_name=run_name)
                        ])
    trainer.fit(model, trainloader, valloader)


if __name__ == "__main__":
    app()
