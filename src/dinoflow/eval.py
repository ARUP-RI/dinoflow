
import logging

import typer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import precision_recall_fscore_support

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
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

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


def train_classifier(backbone, classifier, dataloader, optimizer, val_loader, epochs, tube='m'):
    """
    Train a classifier on the data
    """
    criterion = nn.BCELoss()
    backbone.eval()
    classifier.train()
    for epoch in range(epochs):
        for i, (batch, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            representations = backbone(batch.to(DEVICE).float())
            preds = classifier(representations)
            loss = criterion(preds, labels.to(DEVICE).float().unsqueeze(1))
            logger.info(f"Epoch {epoch} Batch {i} Loss {loss.item() :.4f}")
            
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            allpreds = []
            alllabels = []
            threshold = 0.5
            for i, (batch, labels) in enumerate(val_loader):
                representations = backbone(batch.to(DEVICE).float())
                preds = classifier(representations)
                allpreds.append(preds)
                alllabels.append(labels)
            allpreds = torch.cat(allpreds).cpu().numpy()
            alllabels = torch.cat(alllabels).cpu().numpy()
            precision, recall, fscore, support = precision_recall_fscore_support(alllabels, allpreds > threshold, average='binary')
            logger.info(f"Epoch {epoch} Precision {precision:.4f} Recall {recall:.4f} F-score {fscore:.4f}")
            
            

@app.command()
def train(train_labels, test_labels, checkpoint, events: int = 4096) :
    """
    Evaluate the model on the test set
    """
    model = load_checkpoint(checkpoint).to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    classifier = ClassificationHead(model.cls_token.shape[-1], 1).to(DEVICE)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    traindata = TubeData(train_labels, tubes_to_return=["m"], events_to_return=int(events))
    trainloader = DataLoader(traindata, batch_size=128, shuffle=True)
    logger.info(f"Loaded {len(trainloader.dataset)} samples for training")

    valdata = TubeData(test_labels, tubes_to_return=["m"], events_to_return=int(events))
    valloader = DataLoader(valdata, batch_size=128, shuffle=False)
    logger.info(f"Loaded {len(valloader.dataset)} samples for val")

    train_classifier(model, classifier, trainloader, optimizer, 10, tube='m')


if __name__ == "__main__":
    app()
