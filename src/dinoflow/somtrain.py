
import logging
import os
import yaml
from pathlib import Path
import pickle
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from dinoflow.sombb.som import SOM
from dinoflow import data
import typer


app = typer.Typer(pretty_exceptions_show_locals=False)  

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]  %(name)s  %(levelname)s  %(message)s')

def collate_fn(batch):
    """
    Collate function to be used with DataLoader
    :param batch:
    :return:
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    batch = torch.stack(batch, dim=0)
    labels = torch.zeros(batch.shape[0], dtype=torch.long)
    return labels, batch.flatten(start_dim=0, end_dim=1) # Flatten batch and event dim, result will be [batch_size * evets per sample, 13]

def train_som(conf, run_name):
    """
    Train representation model using SOM 
    :param conf:
    :param run_name:
    :return:
    """

    if 'cuda' in str(DEVICE):
        for idev in range(torch.cuda.device_count()):
            logger.info(f"CUDA device {idev} name: {torch.cuda.get_device_name({idev})}")

    tube_type = conf['tube_type']
    feat_means = torch.tensor(conf['normalization_params'][f"{tube_type}_feat_means"])
    feat_stds = torch.tensor(conf['normalization_params'][f"{tube_type}_feat_stds"])
    transforms = partial(
        data.standardize_range, means=feat_means, stds=feat_stds
    )


    tube_data = data.NoLabelTubes(
        dirpath=conf['data']['data_dir'],
        min_events=conf['data']['input_events'],
        return_key=conf['tube_type'],
        transforms=transforms,
    )

    loader = DataLoader(
        tube_data,
        batch_size=conf['som']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=16,
    )
    logger.info(f"Number of tubes: {len(tube_data)}")
    logger.info(f"Number of batches: {len(loader)}")
    som_conf = conf['som']

    som = SOM(som_conf['size'],
                som_conf['size'], 
                som_conf['n_features'], 
                sched="exp",
                device=DEVICE,
                n_epoch=som_conf['n_epoch']).to(DEVICE)
    for epoch in range(som_conf['n_epoch']):
        learning_error = som.fit(loader, 
                             alpha=2.0,
                             sigma=5.0,
                             print_each=1, 
                             batch_size=som_conf['batch_size'], 
                             num_workers=16, 
                             n_epoch=1)
        logger.info(f"Epoch {epoch}: Learning error: {learning_error}")
        centroids_dict = dict(centroids=som.centroids, m=som_conf['size'], n=som_conf['size'])
        out_name = f'{run_name}_som_centroids_ep{epoch}.p'
        pickle.dump(centroids_dict, open(out_name, "bw"))
    return som, learning_error

@app.command()
def infer(config: Path, ckpt: str):
    with open(config, 'r') as f:
        conf = yaml.safe_load(f)

    # som_conf = conf['som']
    with open(ckpt, mode='rb') as fh:
        centroids_data = pickle.load(fh)
    m, n = centroids_data["m"], centroids_data["n"]
    features = centroids_data['centroids'].shape[1]
    centroids = centroids_data['centroids'].reshape(m, n, features)

    som = SOM(m, n, 
        features, 
        centroids=centroids,
        sched="exp",
        device=DEVICE).to(DEVICE)
    
    # Load data, don't forget to normalize, then project
    
    print(som)
    

@app.command()
def train(config: Path, tube_type: str, run_name: str):

    assert tube_type in ["t", "b", "m"]

    with open(config, 'r') as f:
        conf = yaml.safe_load(f)
    conf['tube_type'] = tube_type
    
    som, learning_error = train_som(conf, run_name)
    logger.info(f"Training finished, learning error: {learning_error}")


if __name__ == "__main__":
    app()
