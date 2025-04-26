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
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CometLogger, CSVLogger
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score

from torch.utils.data import DataLoader, Dataset
import numpy as np


from dinoflow.models import TubeEncoder, TubeEncoderWithProjection, load_checkpoint, BTMTubes, load_btm_from_checkpoint, IlseBagModel, munge_state_dict
from dinoflow.data import TubeData, collate_fn, compose, shift, scale, noise, standardize_range, CSVDataset
from dinoflow import util
from dinoflow.plmodels import BinaryClassificationModel, ClassificationModel, RegressionModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = typer.Typer(pretty_exceptions_show_locals=False)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]   %(levelname)s   %(message)s')

logger = logging.getLogger(__name__)


def load_pl_checkpoint(path, device=None):
    ckpt = torch.load(path, weights_only=False, map_location='cpu')
    backbone_class = ckpt['hyper_parameters']['backbone_class']
    model_conf = ckpt['hyper_parameters']['model_conf']
    num_classes = model_conf.get('num_classes', 1)
    state_dict = munge_state_dict(ckpt['state_dict'])
    if backbone_class == "IlseBagModel":
        model = IlseBagModel(
            num_features=model_conf['num_features'],
            model_embed_dim=model_conf['model_embed_dim'],
            output_classes=model_conf['output_classes'],
            proto_dim=model_conf['proto_dim'],
            bag_classes=model_conf['bag_classes']
        )
    elif backbone_class == "CombinedModel":
        backbone = TubeEncoder(
            num_features=model_conf['num_features'],
            model_embed_dim=model_conf['model_embed_dim'],
            layers=model_conf['layers'],
            heads=model_conf['heads'],
            d_ff=model_conf['d_ff'],
            layertype=model_conf['layertype'],
        )
        classifier = ClassificationHead(model_conf['model_embed_dim'], num_classes, model_conf['output_scale_factor'])
        model = CombinedModel(backbone, classifier)
    else:
        raise NotImplementedError(f"Backbone class {backbone_class} not implemented")
    
    model.load_state_dict(state_dict, strict=True)
    pl_model = ckpt['hyper_parameters']['model_class']
    if pl_model == "BinaryClassificationModel":
        model = BinaryClassificationModel(model, emit_predictions=False)
    elif pl_model == "ClassificationModel":
        model = ClassificationModel(model, num_classes=model_conf['num_classes'], emit_predictions=False)
    elif pl_model == "RegressionModel":
        model = RegressionModel(model, emit_predictions=False)
    else:
        raise NotImplementedError(f"PL model {pl_model} not implemented")
    
    return model

class ClassificationHead(nn.Module):
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
    def __init__(self, backbone, classifier):
        super().__init__()
        self.model_conf = backbone.model_conf
        if hasattr(classifier, 'output_scale_factor'):
            self.model_conf['output_scale_factor'] = classifier.output_scale_factor
        self.backbone = backbone
        self.classifier = classifier
    
    def forward(self, x):
        return self.classifier(self.backbone(x.float()))


class SOMCombinedModel(nn.Module):
    def __init__(self, som, classifier):
        super().__init__()
        self.som = som    
        self.classifier = classifier
    
    def forward(self, x):
        bmus, _, _, _ = self.som.predict(x, num_workers=1, batch_size=10000, print_each=0, return_density=True)
        projection, xi, yi = np.histogram2d(bmus[:, 0], bmus[:, 1], bins=(range(self.som.m + 1), range(self.som.n + 1)))
        normed_projection = projection / np.sum(projection)
        return self.classifier(torch.tensor(normed_projection).float().to(DEVICE))


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
        partial(shift, scale=0.5),
        partial(scale, scale=0.5),
        partial(noise, scale=2.0),
    ])

    val_transforms = compose([
    ])

    # Use the model's specified checkpoint monitor values instead of hardcoding them
    checkpoint_monitor_val = model.checkpoint_monitor.strip()
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
    
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=4)    
    logger.info(f"Loaded {len(trainloader.dataset)} samples for training")

    valdata = TubeData(test_labels, tubes_to_return=tubes, events_to_return=int(events), data_root=dataroot, labelkey=labelkey, transforms=val_transforms)
    valloader = DataLoader(valdata, batch_size=batch_size, shuffle=False, num_workers=4)
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

    checkpoint_dir = f"dinoflow_eval_{run_name}"
    logger.info(f"Checkpoint monitor: {checkpoint_monitor_val}, mode: {checkpoint_monitor_mode}")
    trainer = pl.Trainer(max_epochs=epochs,
                        accelerator='auto',
                        precision="bf16-mixed",
                        callbacks=[
                            ModelCheckpoint(dirpath=checkpoint_dir,
                                            monitor=checkpoint_monitor_val, 
                                            mode=checkpoint_monitor_mode, 
                                            save_top_k=5, 
                                            save_last=True, 
                                            filename=run_name + "_{" + checkpoint_monitor_val + ":.3f}_" + "_{epoch}"),
                            LearningRateMonitor(logging_interval='step'),
                        ],
                        logger=[comet_logger, CSVLogger(save_dir=checkpoint_dir, name=run_name)],
    )

    trainer.fit(model, trainloader, valloader)



@app.command()
def train(run_name, train_labels, test_labels, 
          backbone: str, 
          conf: str, 
          tube_type: str = "", 
          dataroot: str = "/", 
          positive_repeat_factor: int = 1, 
          labelkey: str = "label", 
          checkpoint: str = None, 
          freeze_backbone: bool = False, 
          batch_size: int=16, 
          events: int = 4096, 
          epochs: int = 25, 
          mode: str = 'binary', 
          num_classes: int = 1) :
    """
    Evaluate the model on the test set
    """
    assert tube_type != "", "Tube type must be specified"
    # Helps with too many open files errors?
    torch.multiprocessing.set_sharing_strategy('file_system')

    logger.info(f"Loading backbone from {backbone}")
    backbone, modelconf = load_checkpoint(backbone)
    classifier = ClassificationHead(backbone.cls_token.shape[-1], 
                                    num_classes=1 if mode == 'binary' or mode == 'regression' else num_classes,
                                    output_scale_factor=1.0 if mode == 'regression' else 1.0)
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
                freeze_backbone_layers: int = 0,
                max_lr: float = 0.0001,
                mode:str = 'binary',
                positive_repeat_factor: int = 1,
                num_classes: int = 2):
    # Helps with too many open files errors?
    torch.multiprocessing.set_sharing_strategy('file_system')

    b_backbone, modelconf = load_checkpoint(b_ckpt)
    t_backbone, _ = load_checkpoint(t_ckpt)
    m_backbone, _ = load_checkpoint(m_ckpt)


    output_classes = 1 if mode == 'binary' or mode == 'regression' else num_classes
    output_scale_factor = 2.0 if mode == 'regression' else 1.0

    modelconf['output_classes'] = output_classes # Add it here so it can be saved in the checkpoint

    btm = BTMTubes(num_features=13,
                    model_embed_dim=modelconf['model_dim'],
                    backbone_heads=modelconf['heads'],
                    backbone_layers=modelconf['layers'],
                    output_classes=output_classes,
                    output_scale_factor=output_scale_factor,)

    btm.b_backbone = b_backbone
    btm.t_backbone = t_backbone
    btm.m_backbone = m_backbone

    if freeze_backbone_layers > 0:
        logger.info(f"Freezing backbone layers: {freeze_backbone_layers}")
        for i in range(freeze_backbone_layers):
            btm.b_backbone.encoder.layers[i].eval()
            for p in btm.b_backbone.encoder.layers[i].parameters():
                p.requires_grad = False
            btm.t_backbone.encoder.layers[i].eval()
            for p in btm.t_backbone.encoder.layers[i].parameters():
                p.requires_grad = False
            btm.m_backbone.encoder.layers[i].eval()
            for p in btm.m_backbone.encoder.layers[i].parameters():
                p.requires_grad = False


    if mode == 'binary':
        model = BinaryClassificationModel(btm, emit_predictions=False, ckpt_params=modelconf, max_lr=max_lr)
    elif mode == 'multiclass':
        model = ClassificationModel(btm, num_classes=num_classes, emit_predictions=False, ckpt_params=modelconf, max_lr=max_lr)
    elif mode == 'regression':
        model = RegressionModel(btm, emit_predictions=False, ckpt_params=modelconf, max_lr=max_lr)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    _run_trainer(model, train_labels, test_labels, ["b", "t", "m"], run_name, labelkey, dataroot, events, batch_size, epochs, positive_repeat_factor)
    
    

@app.command()
def continue_training(checkpoint: str,
                     train_labels: str,
                     test_labels: str,
                     run_name: str,
                     model_class: str = None,
                     labelkey: str = "label",
                     dataroot: str = ".",
                     events: int = 4096,
                     freeze_backbone_layers: int = 0,
                     batch_size: int = 16,
                     epochs: int = 50,
                     positive_repeat_factor: int = 1):
    """
    Continue training from a PyTorch Lightning checkpoint.
    
    Args:
        checkpoint: Path to the PyTorch Lightning checkpoint
        train_labels: Path to the training labels CSV
        test_labels: Path to the test labels CSV
        run_name: Name for the run
        labelkey: Column name in the CSV for the labels
        dataroot: Root directory for the data
        events: Number of events to use per sample
        batch_size: Batch size for training
        epochs: Number of epochs to train
        positive_repeat_factor: Factor to repeat positive samples
        tubes: Comma-separated list of tubes to use (e.g., "b,t,m")
        emit_predictions: Whether to emit predictions during validation
    """
    # Helps with too many open files errors?
    torch.multiprocessing.set_sharing_strategy('file_system')

    model, modelconf = load_btm_from_checkpoint(checkpoint, device=DEVICE)
    model.train()
    if freeze_backbone_layers > 0:
        logger.info(f"Freezing backbone layers: {freeze_backbone_layers}")
        for i in range(freeze_backbone_layers):
            model.b_backbone.encoder.layers[i].eval()
            for p in model.b_backbone.encoder.layers[i].parameters():
                p.requires_grad = False
            model.t_backbone.encoder.layers[i].eval()
            for p in model.t_backbone.encoder.layers[i].parameters():
                p.requires_grad = False
            model.m_backbone.encoder.layers[i].eval()
            for p in model.m_backbone.encoder.layers[i].parameters():
                p.requires_grad = False

    assert model_class in ["BinaryClassificationModel", "ClassificationModel", "RegressionModel"], f"Unknown model class: {model_class}"
    mclass = eval(model_class)
    model = mclass(model, emit_predictions=False, ckpt_params=modelconf, num_classes=modelconf['num_classes'])
    
    # Run training
    _run_trainer(model, train_labels, test_labels, ["b", "t", "m"], run_name, 
                labelkey, dataroot, events, batch_size, epochs, positive_repeat_factor)
    
def flatten(x):
    return x.flatten()

def bool_label_transform(x):
    return 1 if x else 0

def flat_and_concat(x):
    if isinstance(x, list):
        y = [torch.tensor(t.flatten()) for t in x]
        x = torch.cat(y, dim=0)
        return x
    else:
        return x.flatten()

@app.command()
def trainsomclassifier(train_csv: str,
                       test_csv: str,
                       run_name: str,
                       mode: str = 'binary',
                       dataroot: str = ".",
                       path_key: str = "path",
                       label_key: str = "label",
                       model_dim: int = 1024,
                       batch_size: int = 16,
                       epochs: int = 50,
                       max_lr: float = 0.0002,
                       num_classes: int = 2,
                       workers: int = 8,
                       positive_repeat_factor: int = 1):
    """
    Train a classifier on projections from a SOM. Really this doesn't know anything about SOMs, the Dataset
    just reads precomputed projections, which I suppose could be from anything.
    """
    
    assert mode in ('binary', 'multiclass', 'regression'), f"Unknown mode: {mode}"
    
    label_transform = float
    if mode == "binary":
        label_transform = bool_label_transform

    # To support 3 tubes, we allow the path_key thing to be a comma-separated list of keys into things in the label CSV
    # Typically this will be something like b_projection,t_projection,m_projection or similar
    # The CSVDataset grabs and loads each element and returns them in a list
    if "," in path_key:
        logger.info(f"Splitting path_key: {path_key}")
        path_key = path_key.split(",")
        model_dim = model_dim * len(path_key)
        logger.info(f"Adjusting model_dim to: {model_dim}")


    clfhead = ClassificationHead(model_dim, 
                                 num_classes=1 if mode == 'binary' or mode == 'regression' else num_classes,
                                 output_scale_factor=0.25 if mode == 'regression' else 1.0)

    traindata = CSVDataset(rootdir=dataroot, csvpath=train_csv, label_key=label_key, path_key=path_key, label_transforms=label_transform, transforms=flat_and_concat)
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=workers)

    valdata = CSVDataset(rootdir=dataroot, csvpath=test_csv, label_key=label_key, path_key=path_key, label_transforms=label_transform, transforms=flat_and_concat)
    valloader = DataLoader(valdata, batch_size=batch_size, shuffle=False, num_workers=workers)

    if mode == 'binary':
        model = BinaryClassificationModel(clfhead, emit_predictions=True, max_lr=max_lr)
    elif mode == 'multiclass':
        model = ClassificationModel(clfhead, num_classes=num_classes, emit_predictions=True, max_lr=max_lr)
    elif mode == 'regression':
        model = RegressionModel(clfhead, emit_predictions=True, max_lr=max_lr)
        assert positive_repeat_factor == 1
    else:
        raise ValueError(f"Unknown mode: {mode}")

    checkpoint_monitor_val = model.checkpoint_monitor.strip()
    checkpoint_monitor_mode = model.checkpoint_mode
    comet_project = model.comet_project

    comet_logger = CometLogger(
            workspace="brendan",  # Optional
            project=comet_project,  # Optional
            name=run_name,  # Optional
            save_dir="dinoflow_classifier_runs",  # Optional
        )
    
    checkpoint_dir = f"dinoflow_{run_name}"
    trainer = pl.Trainer(max_epochs=epochs,
                    accelerator='auto',
                    precision="bf16-mixed",
                    callbacks=[
                        ModelCheckpoint(dirpath=checkpoint_dir,
                                        monitor=checkpoint_monitor_val, 
                                        mode=checkpoint_monitor_mode, 
                                        save_top_k=5, 
                                        save_last=True, 
                                        filename=run_name + "_{" + checkpoint_monitor_val + ":.3f}_" + "_{epoch}"),
                        LearningRateMonitor(logging_interval='step'),
                    ],
                    logger=[comet_logger, CSVLogger(save_dir=checkpoint_dir, name=run_name)])

    trainer.fit(model, trainloader, valloader)

@app.command()
def train_deepcytof(train_csv: str,
                       test_csv: str,
                       run_name: str,
                       tube_type: str,
                       mode: str = 'binary',
                       dataroot: str = ".",
                       label_key: str = "label",
                       batch_size: int = 16,
                       epochs: int = 50,
                       max_lr: float = 0.0002,
                       num_classes: int = 2,
                       events: int = 8192,
                       positive_repeat_factor: int = 1):
    from dinoflow.cnnmodel import DeepCyTof
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    if "," in tube_type:
        tube_type = tube_type.split(",")

    assert mode in ('binary', 'multiclass', 'regression'), f"Unknown mode: {mode}"
    model = DeepCyTof(num_features=13, pool_height=events)
    model = model.to(DEVICE)

    if mode == 'binary':
        model = BinaryClassificationModel(model, emit_predictions=False, max_lr=max_lr)
    elif mode == 'multiclass':
        model = ClassificationModel(model, num_classes=num_classes, emit_predictions=False, max_lr=max_lr)
    elif mode == 'regression':
        model = RegressionModel(model, emit_predictions=False, max_lr=max_lr)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    _run_trainer(model, train_csv, test_csv, tube_type, run_name, label_key, dataroot, events, batch_size, epochs, positive_repeat_factor)
   
@app.command()
def train_abmil(train_csv: str,
                       test_csv: str,
                       run_name: str,
                       tube_type: str,
                       mode: str = 'binary',
                       dataroot: str = ".",
                       label_key: str = "label",
                       batch_size: int = 16,
                       epochs: int = 50,
                       max_lr: float = 0.0002,
                       num_classes: int = 2,
                       events: int = 8192,
                       positive_repeat_factor: int = 1):
    from dinoflow.models import IlseBagModel
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    if "," in tube_type:
        tube_type = tube_type.split(",")

    assert mode in ('binary', 'multiclass', 'regression'), f"Unknown mode: {mode}"
    model = IlseBagModel(13, model_embed_dim=128, output_classes=1, proto_dim=256, bag_classes=4)
    model = model.to(DEVICE)

    if mode == 'binary':
        model = BinaryClassificationModel(model, emit_predictions=False, max_lr=max_lr)
    elif mode == 'multiclass':
        model = ClassificationModel(model, num_classes=num_classes, emit_predictions=False, max_lr=max_lr)
    elif mode == 'regression':
        model = RegressionModel(model, emit_predictions=False, max_lr=max_lr)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    _run_trainer(model, train_csv, test_csv, tube_type, run_name, label_key, dataroot, events, batch_size, epochs, positive_repeat_factor)


@app.command()
def train_abmil3tube(train_csv: str,
                       test_csv: str,
                       run_name: str,
                       mode: str = 'binary',
                       dataroot: str = ".",
                       label_key: str = "label",
                       batch_size: int = 16,
                       epochs: int = 50,
                       max_lr: float = 0.0002,
                       num_classes: int = 2,
                       events: int = 8192,
                       positive_repeat_factor: int = 1):
    from dinoflow.models import Ilse3TubeModel
    torch.multiprocessing.set_sharing_strategy('file_system')

    assert mode in ('binary', 'multiclass', 'regression'), f"Unknown mode: {mode}"
    model = Ilse3TubeModel(13, model_embed_dim=128, output_classes=1, proto_dim=256, bag_classes=1)
    model = model.to(DEVICE)

    if mode == 'binary':
        model = BinaryClassificationModel(model, emit_predictions=False, max_lr=max_lr)
    elif mode == 'multiclass':
        model = ClassificationModel(model, num_classes=num_classes, emit_predictions=False, max_lr=max_lr)
    elif mode == 'regression':
        model = RegressionModel(model, emit_predictions=False, max_lr=max_lr)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    _run_trainer(model, train_csv, test_csv, ["b", "t", "m"], run_name, label_key, dataroot, events, batch_size, epochs, positive_repeat_factor)
   
@app.command()
@torch.inference_mode()
def predict_onetube(checkpoint: str,
                    test_labels: str,
                    tube_type: str,
                    labelkey: str, 
                    dataroot: str = ".", 
                    events: int = 4096, 
                    batch_size: int = 16):
    """
    Predict the labels for the test set
    """
    # In the future we'll be able to load the modelconf from the checkpoint but older models dont save it
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = load_pl_checkpoint(checkpoint, device=device)
    ckpt = torch.load(checkpoint, weights_only=False, map_location='cpu')
    model_conf = ckpt['hyper_parameters']['model_conf']
    num_classes = 1
    backbone = TubeEncoder(
            num_features=model_conf['num_features'],
            model_embed_dim=model_conf['model_embed_dim'],
            layers=model_conf['layers'],
            heads=model_conf['heads'],
            d_ff=model_conf['d_ff'],
            layertype=model_conf['layertype'],
    )
    classifier = ClassificationHead(model_conf['model_embed_dim'], num_classes, model_conf['output_scale_factor'])
    model = CombinedModel(backbone, classifier)
    state_dict = munge_state_dict(ckpt['state_dict'])
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    
    testdata = TubeData(test_labels, data_root=dataroot, labelkey=labelkey, tubes_to_return=[tube_type], events_to_return=int(events))
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=4)
    logger.info(f"Loaded {len(testloader.dataset)} samples for test")

    for b, (batch, rowdict) in enumerate(testloader):
        labels = rowdict['label']
        i = 0
        preds = model(batch.to(device)).cpu()
        num_classes = len(preds.shape) - 1
        if num_classes == 1:
            preds = F.sigmoid(preds) * 100.0
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
    model, modelconf = load_btm_from_checkpoint(checkpoint, device=DEVICE)
    model.eval().to(DEVICE)
    
    testdata = TubeData(test_labels, data_root=dataroot, labelkey=labelkey, tubes_to_return=["b", "t", "m"], events_to_return=int(events))
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=4)
    logger.info(f"Loaded {len(testloader.dataset)} samples for test")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = modelconf['num_classes']
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
