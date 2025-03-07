import logging
logger = logging.getLogger(__name__)

import torch
import typer
import numpy as np
from tqdm import tqdm

from torchmetrics.classification import BinaryF1Score, BinaryRecall, BinaryPrecision, BinaryAccuracy
from pytorch_lightning.loggers import CometLogger

from torch.utils.data import DataLoader

from dinoflow.data import TubeData
from dinoflow.models import load_checkpoint, BTMTubes
from dinoflow.data import compose, shift, scale, noise, standardize_range

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = typer.Typer(pretty_exceptions_show_locals=False)


logging.basicConfig(level=logging.INFO, format='[%(asctime)s]   %(levelname)s   %(message)s')

# Define device globally
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


app = typer.Typer(pretty_exceptions_show_locals=False)

def load_btm_model(b_ckpt, t_ckpt, m_ckpt, output_classes):
    b_backbone, modelconf = load_checkpoint(b_ckpt)
    t_backbone, _ = load_checkpoint(t_ckpt)
    m_backbone, _ = load_checkpoint(m_ckpt)

    b_backbone.eval()
    t_backbone.eval()
    m_backbone.eval()
    
    btm = BTMTubes(num_features=13,
                    model_embed_dim=modelconf['model_dim'],
                    backbone_heads=modelconf['heads'],
                    backbone_layers=modelconf['layers'],
                    output_classes=output_classes)

    btm.b_backbone = b_backbone
    btm.t_backbone = t_backbone
    btm.m_backbone = m_backbone
    
    # Move model to the appropriate device
    btm = btm.to(DEVICE)
    if "cuda" in DEVICE.type:
        btm = torch.nn.DataParallel(btm)
    
    return btm

def fit_knn(model, trainloader):
    """Iterate over the training set and extract the features from the model, then fit a KNN classifier"""
    train_features = []
    train_labels = []

    for i, batch in enumerate(trainloader):
        x, rowinfo = batch
        labels = rowinfo['label']
        logger.info(f"Batch {i} of {len(trainloader)}")
        with torch.no_grad():
            features = model(x)
            # Move features back to CPU for sklearn
            train_features.append(features.float().cpu())
            train_labels.append(labels)

    train_features = torch.cat(train_features, dim=0).float().numpy()
    train_labels = torch.cat(train_labels, dim=0).int().numpy()

    logger.info(f"Training KNN classifier with {len(train_features)} samples")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_features, train_labels)

    return knn


def predict_knn(knn, model, testloader):
    """Iterate over the test set and extract the features from the model, then predict the labels"""
    test_features = []
    test_labels = []

    for i, batch in enumerate(testloader):
        x, rowinfo = batch
        labels = rowinfo['label']
        logger.info(f"Predicting batch {i} of {len(testloader)}")
        with torch.no_grad():
            features = model(x)
            # Move features back to CPU for sklearn
            test_features.append(features.cpu())
            test_labels.append(labels)

    test_features = torch.cat(test_features, dim=0).float().numpy()
    test_labels = torch.cat(test_labels, dim=0).int().numpy()
    preds = knn.predict(test_features)
    # Assess multiclass accuracy, precision, recall, f1 score using sklearn metrics
    acc = accuracy_score(test_labels, preds)
    prec = precision_score(test_labels, preds)
    rec = recall_score(test_labels, preds)
    f1 = f1_score(test_labels, preds)

    return acc, prec, rec, f1
    

@app.command()
def main(run_name, train_labels, test_labels,
          b_ckpt, t_ckpt, m_ckpt,
          num_classes: int = 0,
          labelkey: str = "label",
          dataroot: str = ".",
          batch_size: int = 16,
          events: int = 4096,
          ) :
    """
    Evaluate the model on the test set
    """
    assert num_classes > 1, "num_classes must be greater than 1"
    
    logger.info(f"Using device: {DEVICE}")
    model = load_btm_model(b_ckpt, t_ckpt, m_ckpt, 1)

    torch.set_float32_matmul_precision('medium')
    
    traindata = TubeData(train_labels, events_to_return=int(events), labelkey=labelkey, data_root=dataroot)
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=16)
    logger.info(f"Loaded {len(trainloader.dataset)} samples for training")
    logger.info(f"Positive samples: {len(traindata.positive_negative_samples()[0])}")
    logger.info(f"Negative samples: {len(traindata.positive_negative_samples()[1])}")

    valdata = TubeData(test_labels, events_to_return=int(events), labelkey=labelkey, data_root=dataroot)
    valloader = DataLoader(valdata, batch_size=batch_size, shuffle=False, num_workers=16)
    logger.info(f"Loaded {len(valloader.dataset)} samples for val")
    logger.info(f"Positive samples: {len(valdata.positive_negative_samples()[0])}")
    logger.info(f"Negative samples: {len(valdata.positive_negative_samples()[1])}")

    knn = fit_knn(model, trainloader)

    acc, prec, rec, f1 = predict_knn(knn, model, testloader=valloader)
    logger.info(f"Accuracy: {acc}")
    logger.info(f"Precision: {prec}")
    logger.info(f"Recall: {rec}")
    logger.info(f"F1: {f1}")


if __name__ == "__main__":
    app()
