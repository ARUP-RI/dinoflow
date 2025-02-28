
import logging
from functools import partial

import typer
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import BinaryPrecisionRecallCurve, BinaryF1Score, BinaryRecall, BinaryPrecision, BinaryAccuracy
from torchmetrics.aggregation import MeanMetric, SumMetric

from torch.utils.data import DataLoader
import numpy as np


from dinoflow.models import TubeEncoder, TubeEncoderWithProjection, load_checkpoint
from dinoflow.data import TubeData, collate_fn, compose, shift, scale, noise, standardize_range
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
    def __init__(self, backbone, classifier, min_lr=0.00001, max_lr=0.0001, warmup_iters=20, lr_decay_iters=250):
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
        self.precision25 = BinaryPrecision()  
        self.recall25 = BinaryRecall()
        self.f1score25 = BinaryF1Score()

        self.training_loss_mean = MeanMetric()
        self.validation_loss_mean = MeanMetric()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, labels = batch
        preds = self(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(preds.squeeze(1), labels.float())
        self.training_loss_mean.update(loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, labels = batch
        preds = self(x).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(preds, labels.float())
        preds = torch.nn.Sigmoid()(preds) # raw outputs are logits, non-sigmoid
        #for p, l in zip(preds, labels):
        #    print(f"{p.item() :.4f}\t{l.item() :.2f}")
        self.accuracy(preds, labels)
        self.precision(preds, labels)
        self.recall(preds, labels)
        self.f1score(preds, labels)
        self.precision25(preds > 0.25, labels)
        self.recall25(preds > 0.25, labels)
        self.f1score25(preds > 0.25, labels)
        
        self.validation_loss_mean.update(loss)

    def on_validation_epoch_end(self):
        lrsched = self.lr_schedulers()
        lr = lrsched.get_last_lr()[0]
        accuracy = self.accuracy.compute()
        precision = self.precision.compute()
        recall = self.recall.compute()
        fscore = self.f1score.compute()
        precision25 = self.precision25.compute()
        recall25 = self.recall25.compute()
        fscore25 = self.f1score25.compute()

        self.log('precision', precision, sync_dist=True )
        self.log('accuracy', accuracy, sync_dist=True)
        self.log('recall', recall, sync_dist=True)
        self.log('fscore', fscore, sync_dist=True)
        self.log('val_loss', self.validation_loss_mean.compute(), sync_dist=True)
        self.log('training_loss', self.training_loss_mean.compute(), sync_dist=True)
        self.log('precision25', precision25, sync_dist=True)
        self.log('recall25', recall25, sync_dist=True)
        self.log('fscore25', fscore25, sync_dist=True)
        self.log('learning_rate', lr)
    
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1score.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        lrschedule = util.WarmupCosineLRScheduler(optimizer, self.max_lr, self.min_lr, self.warmup_iters, self.lr_decay_iters)
        return [optimizer], [lrschedule]



def load_featmeans_stds(conf, tube_type):
    if tube_type == 't':
        return conf['normalization_params']['t_feat_means'], conf['normalization_params']['t_feat_stds']
    elif tube_type == 'm':
        return conf['normalization_params']['m_feat_means'], conf['normalization_params']['m_feat_stds']
    elif tube_type == 'b':
        return conf['normalization_params']['b_feat_means'], conf['normalization_params']['b_feat_stds']
    else:
        raise ValueError(f"Unknown tube type: {tube_type}")


@app.command()
def train(run_name, train_labels, test_labels, backbone: str, conf: str, dataroot: str = "/", labelkey: str = "label", checkpoint: str = None, freeze_backbone: bool = False, batch_size: int=16, events: int = 4096, epochs: int = 25, tube_type: str = "m") :
    """
    Evaluate the model on the test set
    """
    logger.info(f"Loading backbone from {backbone}")
    backbone = load_checkpoint(backbone, device=DEVICE)
    classifier = ClassificationHead(backbone.cls_token.shape[-1], 1)
    model = ClassificationModel(backbone, classifier)

    with open(conf, 'r') as f:
        conf = yaml.safe_load(f)

    feat_means, feat_stds = load_featmeans_stds(conf, tube_type)
    feat_means = torch.tensor(feat_means).to(model.device)
    feat_stds = torch.tensor(feat_stds).to(model.device)

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

    train_transforms = compose([
        partial(shift, scale=0.1),
        partial(scale, scale=0.1),
        partial(noise, scale=0.25),
    ])

    val_transforms = compose([
    ])
    
    traindata = TubeData(train_labels, tubes_to_return=[tube_type], events_to_return=int(events), dataroot=dataroot, labelkey=labelkey, transforms=train_transforms)
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=16)
    logger.info(f"Loaded {len(trainloader.dataset)} samples for training")
    logger.info(f"Positive samples: {len(traindata.positive_negative_samples()[0])}")
    logger.info(f"Negative samples: {len(traindata.positive_negative_samples()[1])}")

    valdata = TubeData(test_labels, tubes_to_return=[tube_type], events_to_return=int(events), dataroot=dataroot, labelkey=labelkey, val_transforms=val_transforms)
    valloader = DataLoader(valdata, batch_size=batch_size, shuffle=False, num_workers=16)
    logger.info(f"Loaded {len(valloader.dataset)} samples for val")
    logger.info(f"Positive samples: {len(valdata.positive_negative_samples()[0])}")
    logger.info(f"Negative samples: {len(valdata.positive_negative_samples()[1])}")


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
                            ModelCheckpoint(dirpath=f"dinoflow_eval_{run_name}", monitor='fscore', mode='max', save_top_k=1, save_last=True, filename=run_name + "_e{epoch}"),
                            LearningRateMonitor(logging_interval='step'),
                        ],
                        logger=comet_logger)

    trainer.fit(model, trainloader, valloader)



@app.command()
def predict(checkpoint, test_labels, events: int = 4096, batch_size: int = 16):
    """
    Predict the labels for the test set
    """
    logger.info(f"Loading checkpoint from {checkpoint}")
    backbone_conf = {
        "num_features": 13,
        "model_dim": 512,
        "layers": 10,
        "heads": 4,
        "hidden_dim": 256,
        "projection_dim": 256
    }
    backbone = TubeEncoder(num_features=backbone_conf['num_features'],
                                        model_embed_dim=backbone_conf['model_dim'],
                                        layers=backbone_conf['layers'],
                                        heads=backbone_conf['heads']).to(DEVICE)
    classifier = ClassificationHead(backbone.cls_token.shape[-1], 1)
    model = ClassificationModel.load_from_checkpoint(checkpoint, backbone=backbone, classifier=classifier)  
    model.eval()
    testdata = TubeData(test_labels, tubes_to_return=["m"], events_to_return=int(events))
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=16)
    logger.info(f"Loaded {len(testloader.dataset)} samples for test")

    with torch.inference_mode():
        for b, (batch, labels) in enumerate(testloader):
            i = 0
            preds = model(batch)
            preds = F.sigmoid(preds)
            for p, l in zip(preds, labels):
                rowdata = testdata.get_row_data(b+i)
                print(f"{rowdata['accession']}\t {p.item() :.4f}\t {l.item()}")
                i += 1

if __name__ == "__main__":
    app()
