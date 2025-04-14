
import logging
import os
import yaml
from pathlib import Path
import pickle
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dinoflow.sombb.quicksom import SOM
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
    return labels.cpu().numpy(), batch.flatten(start_dim=0, end_dim=1).cpu().numpy() # Flatten batch and event dim, result will be [batch_size * evets per sample, 13]


def init_centroids(dataset, m, n):
    """
    For each point in the m * n grid, pick a random sample from the dataset, then pick a random event from that sample, and set the
    centroid to that event.
    """
    centroids = np.zeros((m * n, dataset[0].shape[1]))
    for i in range(m):
        for j in range(n):
            # Pick a random sample from the dataset
            sample = dataset[np.random.randint(0, len(dataset))]
            # Pick a random event from that sample
            event = sample[np.random.randint(0, len(sample))]
            centroids[i * n + j] = event.cpu().numpy()

    return centroids


def train_som(conf, run_name):
    """
    Train representation model using SOM 
    :param conf:
    :param run_name:
    :return:
    """

    # Torch says this fixes a "too many open files" issue
    torch.multiprocessing.set_sharing_strategy('file_system')

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
        batch_size=conf['som']['batch_size'], # batch size here means number of tubes (each tube will have many events)
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    logger.info(f"Number of tubes: {len(tube_data)}")
    logger.info(f"Number of batches: {len(loader)}")
    som_conf = conf['som']

    centroids = init_centroids(tube_data, som_conf['size'], som_conf['size'])

    output_dir = Path(f"{run_name}_som_checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_dir, f"{run_name}_som_config.yaml"), 'w') as f:
        yaml.dump(conf, f)

    os.chdir(output_dir)
    som = SOM(som_conf['size'],
                som_conf['size'], 
                som_conf['n_features'], 
                sched="exp",
                centroids=centroids,
                n_epoch=som_conf['n_epoch'])
    alpha = som_conf['alpha']
    
    learning_error = som.fit(loader, 
                            alpha=alpha,
                            sigma=som_conf['sigma'],
                            print_each=10,
                            start_epoch=0, 
                            checkpoint_freq=1,
                            checkpoint_name_base=f"{run_name}_som",
                            batch_size=som_conf['batch_size'], 
                            num_workers=1, 
                            n_epoch=som_conf['n_epoch'])
    # logger.info(f"Epoch {epoch}: Learning error: {learning_error}")
    # centroids_dict = dict(centroids=som.centroids, m=som_conf['size'], n=som_conf['size'])
    # out_name = f'{run_name}_som_centroids_ep{epoch}.p'
    # pickle.dump(centroids_dict, open(out_name, "bw"))
    return som, learning_error

@app.command()
def infer(config: str, tubedata: str, tube_type: str, ckpt: str, destdir: str = "."):
    assert tube_type in ("b", "t", "m")
    
    with open(config, 'r') as f:
        conf = yaml.safe_load(f)

    feat_means = torch.tensor(conf['normalization_params'][f"{tube_type}_feat_means"])
    feat_stds = torch.tensor(conf['normalization_params'][f"{tube_type}_feat_stds"])
    transforms = partial(
        data.standardize_range, means=feat_means, stds=feat_stds
    )

    destdir = Path(destdir)
    destdir.mkdir(parents=True, exist_ok=True)

    with open(ckpt, mode='rb') as fh:
        centroids_data = pickle.load(fh)
    m, n = centroids_data["m"], centroids_data["n"]
    features = centroids_data['centroids'].shape[1]
    centroids = centroids_data['centroids'].reshape(m * n, features)

    data_paths = []
    if tubedata.endswith(".pt"):
        data_paths.append(tubedata)
        
    elif Path(tubedata).is_dir():
        data_paths.extend(Path(tubedata).glob("*.pt"))

    else:
        raise ValueError(f"Invalid tubedata: {tubedata}")
        

    som = SOM(m, n, features, centroids=centroids)
    
    for i, path in enumerate(data_paths):
        sample_name = Path(path).stem
        logger.info(f"Processing sample {i + 1}/{len(data_paths)}: {sample_name}")

        td = torch.load(path, weights_only=False)
        dataset = data.SimpleTubeSet([td[tube_type][0:100000,:]], transforms=transforms)
        loader = DataLoader(dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=1,
        )
        # Load data, don't forget to normalize, then project
        bmus, errs, labels = som.predict(loader, num_workers=1, batch_size=64000, print_each=0)
        dim1_data = bmus[:, 0]
        dim2_data = bmus[:, 1]
        projection, xi, yi = np.histogram2d(dim1_data, dim2_data, bins=(range(m + 1), range(n + 1)))
        normed_projection = projection / np.max(projection)
        for i in range(m):
            for j in range(n):
                print(f"{int(100 * normed_projection[i, j]) :3d}", end=" ")
            print()
        np.save(f"{destdir}/{sample_name}_projection.npy", projection)

    

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
