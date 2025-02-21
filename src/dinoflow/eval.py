
import logging

import typer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dinoflow.models import TubeEncoderWithProjection
from dinoflow.data import TubeData, collate_fn
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
def train_classifier(backbone, classifier, dataloader, optimizer, epochs, tube='m'):
    """
    Train a classifier on the data
    """
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            print(f"Batch: {batch}")
            optimizer.zero_grad()
            batch = torch.stack(batch[tube])
            representations = backbone(batch)
            preds = classifier(representations)
            loss = criterion(preds, labels)
            logger.info(f"Epoch {epoch} Batch {i} Loss {loss.item() :.4f}")
            
            loss.backward()
            optimizer.step()

@app.command()
def train(train_labels, test_labels, checkpoint) :
    """
    Evaluate the model on the test set
    """
    model = load_checkpoint(checkpoint)
    for p in model.parameters():
        p.requires_grad = False

    classifier = ClassificationHead(model.cls_token.shape[-1], 2)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    traindata = TubeData(train_labels, tubes_to_return=["m"])
    trainloader = DataLoader(traindata, batch_size=128, shuffle=True, collate_fn=collate_fn)
    logger.info(f"Loaded {len(trainloader.dataset)} samples for training")

    valdata = TubeData(test_labels, tubes_to_return=["m"])
    valloader = DataLoader(valdata, batch_size=128, shuffle=False, collate_fn=collate_fn)
    logger.info(f"Loaded {len(valloader.dataset)} samples for val")

    train_classifier(model, classifier, trainloader, optimizer, 10, tube='m')


if __name__ == "__main__":
    app()
