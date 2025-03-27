
import logging
import os
import yaml
from pathlib import Path

import numpy as np
import torch
from quicksom.som import SOM

from dinoflow import data
import typer


app = typer.Typer(pretty_exceptions_show_locals=False)  

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]  %(name)s  %(levelname)s  %(message)s')


def train_som(conf):
    """
    Train representation model using SOM 
    :param conf:
    :param run_name:
    :return:
    """

    if 'cuda' in str(DEVICE):
        for idev in range(torch.cuda.device_count()):
            logger.info(f"CUDA device {idev} name: {torch.cuda.get_device_name({idev})}")


    tube_data = data.NoLabelTubes(
        dirpath=conf['data']['data_dir'],
        min_events=conf['data']['input_events'] * 4,
        return_key=conf['tube_type'],
    )

    som_conf = conf['som']

    som = SOM(som_conf['size'], som_conf['size'], som_conf['n_features'], n_epoch=som_conf['n_epoch'])
    learning_error = som.fit(tube_data, batch_size=som_conf['batch_size'])
    return som, learning_error


def main(config: Path, run_name: str):

    with open(config, 'r') as f:
        conf = yaml.safe_load(f)
    
    som, learning_error = train_som(conf)

    som.save_pickle(f'{run_name}_som.p')


if __name__ == "__main__":
    app()
