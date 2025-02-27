import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from glob import glob
from functools import partial
from collections import defaultdict
from typing import List
import logging

logger = logging.getLogger(__name__)

from dinoflow import util

def shuffle(x):
    """
    Shuffle the elements on dimension 0 or 1. If the tensor has three dimensions, we shuffle dim 1
    but if it only has two dimensions, we shuffle 0
    Note that if x has 3 dims (presumably because it has shape [batch, events, features]) we shuffle
    the events all the same way across batches. This seems OK, probably, but maybe there's a reason to shuffle
    each batch element separately? It might be a little slower
    """
    if len(x.shape) == 3:
        idx = torch.randperm(x.shape[1])
        return x[:, idx, :]
    elif len(x.shape) == 2:
        idx = torch.randperm(x.shape[0])
        return x[idx, :]
    raise ValueError(f"Input tensor must have 2 or 3 dimensions (found {len(x.shape)})")



def standardize_range(x, means, stds):
    return (x - means) / stds


def normalize(x):
    mean = x.mean(dim=1)
    std = x.std(dim=1)
    return (x - mean.unsqueeze(1).expand(-1, x.shape[1], -1)) / std.unsqueeze(1).expand(-1, x.shape[1], -1)


def bootstrap_events(x, prob=0.5, bootstrap_frac=0.5):
    """
    Randomly replace a fraction of the events with other randomly chosen events
    Modifies in place! Use clone if you need the original tensor unmodified
    """
    r = torch.rand(x.shape[0])
    for w in range(x.shape[0]):
        if r[w] < prob:
            num_to_replace = int(x.shape[1] * bootstrap_frac)
            i = torch.randperm(x.shape[1])
            replace_idx = i[0:num_to_replace]
            remain_idx = i[num_to_replace:]
            # Each replace_idx gets a number randomly chosen from remain_idx
            rw = torch.randint(remain_idx.shape[0], (num_to_replace,)) #
            x[w, replace_idx, :] = x[w, remain_idx[rw], :]
    return x


def subsample_events(x, num_events):
    """
    Select a random sample of events and return those
    Unlike other augmentors, this does the same thing to every sample
    :param x:
    :param num_events: Number of events to subsample
    :return:
    """
    if len(x.shape) == 3:
        ev = torch.randperm(x.shape[1])[0:num_events]
        return x[:, ev, :]
    elif len(x.shape) == 2:
        ev = torch.randperm(x.shape[0])[0:num_events]
        return x[ev, :]
    else:
        raise ValueError(f"Input tensor must have 2 or 3 dimensions (found {len(x.shape)})")
    

def subsample_batch(x: List[torch.Tensor], num_events: int):
    """
    Select a random sample of events for each item in x stack them into a tensor
    If an item has less than num_events, it is skipped
    """
    subsampled_x = []
    for t in x:
        if t.shape[0] < num_events:
            continue
        else:
            subsampled_x.append(subsample_events(t, num_events))

    return torch.stack(subsampled_x, dim=0)


def scale(x, device='cpu', prob=0.5, scale=0.1):
    """ Multiply all events in a channel by the same amount, scaling them all to be a bit bigger or smaller """
    r = torch.rand(x.shape[0])
    for i in range(x.shape[0]):
        if r[i] < prob:
            z = torch.normal(mean=torch.tensor([1.0 for _ in range(x.shape[-1])]),
                             std=torch.tensor([scale for _ in range(x.shape[-1])])).to(device)
            z = torch.clamp(z, min=1.0 - 2*scale, max=1.0 + 2*scale)
            x[i, :, :] = x[i, :, :] * z

    return x

def shift(x, device='cpu', prob=0.5, scale=0.1):
    """
    Add / subtract a constant value to all events in each channel, 'shifts' all channel values up or down a bit
    """
    r = torch.rand(x.shape[0])
    for i in range(x.shape[0]):
        if r[i] < prob:
            z = torch.normal(mean=torch.tensor([0.0 for _ in range(x.shape[-1])]), std=torch.tensor([scale for _ in range(x.shape[-1])])).to(device)
            z = torch.clamp(z, min=-2*scale, max=2*scale)
            x[i, :, :] = x[i, :, :] + z

    return x


def noise(x, device='cpu', prob=0.5, scale=0.1):
    """
    Adds gaussian random noise to each event, individually
    """
    r = torch.rand(x.shape[0])
    for i in range(x.shape[0]):
        if r[i] < prob:
            z = torch.normal(mean=torch.zeros_like(x[i, :, :]), std=(scale * torch.ones_like(x[i, :, :]))).to(device)
            z = torch.clamp(z, min=-2*scale, max=2*scale)
            x[i, :, :] = x[i, :, :] + z

    return x

def compose(funcs):
    """
    Return a new function that composes the given list of functions
    :param funcs: List of callables
    :return: Single callable
    """

    def f(x):
        for func in funcs:
            x = func(x)
        return x
    return f


def onepointoh():
    """ Loaders need to be pickle-able, so they can't contain lambdas, but if we just define a top-level function it works """
    return 1.0

class BTMTubes(Dataset):

    def __init__(self, dirpath,
                 b_transforms=[],
                 t_transforms=[],
                 m_transforms=[],
                 return_index=False,
                 labels_df: pd.DataFrame = None,
                 labels_to_return=None,
                 sample_type_filter=None,
                 min_events=16384,
                 weights={}):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.labels_df = labels_df
        self.b_transforms = b_transforms
        self.t_transforms = t_transforms
        self.m_transforms = m_transforms
        self.weight_dict = defaultdict(onepointoh)
        self.weight_dict.update(weights)
        self.paths = []
        self.labels = []
        self.accs = []
        self.sample_weights = []
        self.return_index = return_index
        self.keys = labels_to_return
        self.min_events = min_events
        self.sample_type_filter = sample_type_filter
        self._scandir()

    def _scandir(self):
        missing_accs = 0
        not_enough_events = 0
        assert self.keys is not None and len(self.keys) > 0, "Provide a label_key to query the DF"

        if self.sample_type_filter is not None:
            labels_df = self.labels_df[self.labels_df['sample_type'] == self.sample_type_filter]
            logger.info(f"Found {len(labels_df)} samples with type {self.sample_type_filter}")

        for path in self.dirpath.iterdir():
            acc = Path(path).name.split("_")[0]
            hits = self.labels_df.query(f"accession == '{acc}'")
            if len(hits) == 0:
                # logger.warning(f"Couldn't find accession {acc} in input DF :(")
                missing_accs += 1
                continue
            else:
                try:
                    x = torch.load(path, map_location='cpu')
                except Exception as ex:
                    logger.error(f"Error reading file: {path} : {ex}")
                    # raise ex
                if x['t'].shape[0] < self.min_events or \
                        x['b'].shape[0] < self.min_events or \
                        x['m'].shape[0] < self.min_events:
                    not_enough_events += 1
                    continue

                labs = hits[self.keys].values[0].tolist()
                self.labels.append(labs)
                self.sample_weights.append(max(self.weight_dict[k] for k,l in zip(hits.columns, hits.values[0]) if l ))
                self.paths.append(path)
                self.accs.append(acc)

        if missing_accs:
            logger.warning(f"Couldn't find {missing_accs} accessions in label DF :(")
        if not_enough_events:
            logger.warning(f"Found {not_enough_events} samples with not enough events")

    def __getitem__(self, i):
        try:
            item = torch.load(self.paths[i])
        except Exception as ex:
            raise Exception(f"Failed to load item {self.paths[i]}: {str(ex)}")

        bx = item['b']
        for tr in self.b_transforms:
            bx = tr(bx)

        tx = item['t']
        for tr in self.t_transforms:
            tx = tr(tx)

        mx = item['m']
        for tr in self.m_transforms:
            mx = tr(mx)

        if self.labels:
            d = dict((k, v) for k,v in zip(self.keys, self.labels[i]))
            d['weight'] = self.sample_weights[i]
            if self.return_index:
                return i, d, [bx, tx, mx]
            else:
                return d, [bx, tx, mx]

        if self.return_index:
            return i, [bx, tx, mx]
        else:
            return [bx, tx, mx]

    def __len__(self):
        return len(self.paths)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def accession(self, i):
        return Path(self.paths[i]).name.split("_")[0]

    def itempath(self, i):
        return Path(self.paths[i])


class NoLabelTubes(Dataset):

    def __init__(self, dirpath, min_events=2048, return_key="t"):
        self.dirpath = Path(dirpath)
        self.min_events = min_events
        self.samples = self._scandir()
        self.return_key = return_key

    def _scandir(self):
        samples = []
        for path in self.dirpath.iterdir():
            try:
                if path.is_file() and path.suffix == '.pt':
                    samples.append(path)
            except Exception as ex:
                logger.error(f"Error reading file: {path} : {ex}")
                # raise ex
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        try:
            item = torch.load(self.samples[i], weights_only=False)
            if item[self.return_key].shape[0] > self.min_events:
                idx = torch.randperm(item[self.return_key].shape[0])[0:self.min_events]
                return item[self.return_key][idx]
            else:
                return item[self.return_key]
        except Exception as ex:
            raise Exception(f"Failed to load item {self.samples[i]}: {str(ex)}")
    
    def get_path(self, i):
        return self.samples[i]
        


class TubeData(Dataset):

    def __init__(self, labelcsv, events_to_return=-1, data_root="/", tubes_to_return=["b", "t", "m"], labelkey="label", transforms=None):
        self.data = pd.read_csv(labelcsv)
        self.tubes_to_return = tubes_to_return
        self.dataroot = Path(data_root)
        self.events_to_return = events_to_return
        self.labelkey = labelkey
        self.transforms = transforms

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        row = self.data.iloc[i]
        tubes = {}
        for tube in self.tubes_to_return:
            tubedata = torch.load(self.dataroot / row['path'], weights_only=False)
            if self.events_to_return != -1:
                tubes[tube] = subsample_events(tubedata[tube], self.events_to_return)
            else:
                tubes[tube] = tubedata[tube]
        
        if self.transforms:
            for tube in self.tubes_to_return:
                tubes[tube] = self.transforms(tubes[tube])
        
        if len(tubes) == 1:
            tubes = tubes[self.tubes_to_return[0]]
         
        # For now....
        label = row[self.labelkey]
        if label:
            label = 1.0
        else:
            label = 0.0

        return tubes, label
    
    def get_row_data(self, i):
        row = self.data.iloc[i]
        return row.to_dict()
    
    def positive_negative_samples(self):
        pos = self.data[self.data[self.labelkey] == 1]
        neg = self.data[self.data[self.labelkey] == 0]
        return pos, neg


def collate_fn(items):
    """
    The NoLabelTubes dataset returns a list of tensors which may have different sizes, so we can't stack them
    """
    return items
