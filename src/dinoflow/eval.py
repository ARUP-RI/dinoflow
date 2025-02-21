
import logging

import typer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dinoflow.models import TubeEncoderWithProjection
from dinoflow.data import TubeData
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
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)

def load_checkpoint(path):
    """
    Load a checkpoint from a file
    """
    ckpt = torch.load(path, weights_only=False, map_location=DEVICE)
    modelconf = ckpt['modelconf']    
    teacher = TubeEncoderWithProjection(num_features=modelconf['num_features'], model_embed_dim=modelconf['model_dim'], layers=modelconf['layers'], heads=modelconf['heads'], hidden_dim=modelconf['hidden_dim'], projection_dim=modelconf['projection_dim']).to(DEVICE)

    teacher.load_state_dict(ckpt['teacher'])
    teacher.eval()
    return teacher.tube_encoder


@torch.inference_mode()
def train_classifier(backbone, classifier, dataloader, optimizer, epochs):
    """
    Train a classifier on the data
    """
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for i, (labels, batch) in enumerate(dataloader):
            optimizer.zero_grad()
            representations = backbone(batch)
            preds = classifier(representations)
            loss = criterion(preds, labels)
            logger.info(f"Epoch {epoch} Batch {i} Loss {loss.item() :.4f}")
            
            loss.backward()
            optimizer.step()


def main(train_labels, test_labels, checkpointpath):
    """
    Evaluate the model on the test set
    """
    model = load_checkpoint(checkpointpath)
    classifier = ClassificationHead(model.num_features, 2)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    # Load the data
    traindata = TubeData(train_labels)
    trainloader = DataLoader(traindata, batch_size=128, shuffle=True)


    valdata = TubeData(test_labels)
    valloader = DataLoader(valdata, batch_size=128, shuffle=False)
    
    train_classifier(model, classifier, trainloader, optimizer, 10)


if __name__ == "__main__":
    app()
