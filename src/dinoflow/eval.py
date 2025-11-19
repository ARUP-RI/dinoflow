import comet_ml  # Must be FIRST import
import logging
from functools import partial

import typer
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CometLogger, CSVLogger
from pytorch_lightning.strategies import DDPStrategy

from torch.utils.data import DataLoader

from dinoflow.models import load_checkpoint, BTMTubes
from dinoflow.data import TubeData, compose, shift, scale, noise
from dinoflow.plmodels import BinaryClassificationModel, ClassificationModel, RegressionModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = typer.Typer(pretty_exceptions_show_locals=False)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]   %(levelname)s   %(message)s')

logger = logging.getLogger(__name__)


def _run_trainer(model, train_labels, test_labels, tubes, run_name, labelkey, dataroot, events, batch_size, epochs, comet_workspace, comet_project, positive_repeat_factor=1):
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
        partial(shift, prob=0.5, scale=0.5),
        partial(scale, prob=0.5, scale=0.5),
        partial(noise, prob=0.5, scale=2.0),
    ])

    val_transforms = compose([
    ])

    # Use the model's specified checkpoint monitor values instead of hardcoding them
    checkpoint_monitor_val = model.checkpoint_monitor.strip()
    checkpoint_monitor_mode = model.checkpoint_mode
    
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

    val_events = 1 * int(events)
    logger.info(f"Train events: {int(events)}, val events: {val_events}")
    valdata = TubeData(test_labels, tubes_to_return=tubes, events_to_return=val_events, data_root=dataroot, labelkey=labelkey, transforms=val_transforms)
    valloader = DataLoader(valdata, batch_size=batch_size, shuffle=False, num_workers=4)
    logger.info(f"Loaded {len(valloader.dataset)} samples for val")
    
    # Check if we have any positive samples for classification models in the validation set
    if hasattr(valdata, 'positive_negative_samples'):
        logger.info(f"Positive samples: {len(valdata.positive_negative_samples()[0])}")
        logger.info(f"Negative samples: {len(valdata.positive_negative_samples()[1])}")
        #assert len(valdata.positive_negative_samples()[0]) > 0, f"No positive samples found :("
    
    comet_logger = CometLogger(
            workspace=comet_workspace if comet_workspace is not None else "r-i",  # Optional
            save_dir="dinoflow_classifier_runs",  # Optional 
            project_name=comet_project if comet_project is not None else "no-name-project",  # Optional
            experiment_name=run_name,  # Optional
        )


    checkpoint_dir = f"dinoflow_eval_{run_name}"
    logger.info(f"Checkpoint monitor: {checkpoint_monitor_val}, mode: {checkpoint_monitor_mode}")
    trainer = pl.Trainer(max_epochs=epochs,
                        accelerator='auto',
                        precision="bf16-mixed",
                        strategy=DDPStrategy(find_unused_parameters=True),
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
def train3tubes(run_name, train_labels, test_labels,
                backbone_b: str, 
                backbone_t: str, 
                backbone_m: str,
                dataroot: str = "/",
                positive_repeat_factor: int = 1,
                labelkey: str = "label",
                freeze_backbone: bool = False,
                freeze_backbone_layers: int = 0,
                batch_size: int = 16,
                events: int = 4096,
                epochs: int = 50,
                mode: str = 'binary',
                num_classes: int = 1,
                max_lr: float = 0.0001,
                comet_workspace: str = None,
                comet_project: str = None,
                pos_weight: float = 1.0,
                checkpoint_monitor: str = 'val_loss',
                checkpoint_mode: str = 'min',):
    # Helps with too many open files errors?
    torch.multiprocessing.set_sharing_strategy('file_system')

    b_backbone, modelconf = load_checkpoint(backbone_b)
    t_backbone, _ = load_checkpoint(backbone_t)
    m_backbone, _ = load_checkpoint(backbone_m)


    output_classes = 1 if mode == 'binary' or mode == 'regression' else num_classes
    output_scale_factor = 2.0 if mode == 'regression' else 1.0

    modelconf['output_classes'] = output_classes # Add it here so it can be saved in the checkpoint

    btm = BTMTubes(num_features=13,
                    model_embed_dim=modelconf['model_dim'],
                    backbone_heads=modelconf['heads'],
                    backbone_layers=modelconf['layers'],
                    output_classes=output_classes,
                    d_ff=modelconf['d_ff'],
                    layer_type=modelconf['layer_type'],
                    dropout=0.0,
                    output_scale_factor=output_scale_factor,)

    btm.b_backbone = b_backbone
    btm.t_backbone = t_backbone
    btm.m_backbone = m_backbone


    if mode == 'binary':
        model = BinaryClassificationModel(btm, emit_predictions=False, ckpt_params=modelconf, max_lr=max_lr, comet_project_name=comet_project, pos_weight=pos_weight, checkpoint_monitor=checkpoint_monitor, checkpoint_mode=checkpoint_mode)
    elif mode == 'multiclass':
        model = ClassificationModel(btm, num_classes=num_classes, emit_predictions=False, ckpt_params=modelconf, max_lr=max_lr, comet_project_name=comet_project)
    elif mode == 'regression':
        model = RegressionModel(btm, emit_predictions=False, ckpt_params=modelconf, max_lr=max_lr, comet_project_name=comet_project)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    if freeze_backbone_layers > 0 and not freeze_backbone:
        raise ValueError("freeze_backbone_layers > 0 requires freeze_backbone to be True")
    if freeze_backbone:
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
        else:
            logger.info("Freezing full backbone")
            btm.b_backbone.eval()
            for p in btm.b_backbone.parameters():
                p.requires_grad = False
            btm.t_backbone.eval()
            for p in btm.t_backbone.parameters():
                p.requires_grad = False
            btm.m_backbone.eval()
            for p in btm.m_backbone.parameters():
                p.requires_grad = False
    else:
        logger.info("Unfreezing backbone")
        b_backbone.train()
        t_backbone.train()
        m_backbone.train()
   
    _run_trainer(model, train_labels, test_labels, ["b", "t", "m"], run_name, labelkey, dataroot, events, batch_size, epochs, comet_workspace, comet_project, positive_repeat_factor)
    

@app.command()
def predict(checkpoint: str,
            test_labels: str,
            labelkey:str, 
            dataroot: str = ".", 
            events: int = 16384, 
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
