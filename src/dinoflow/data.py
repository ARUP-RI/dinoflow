import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from glob import glob
from functools import partial
import numpy as np
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


#def bootstrap_events(x, prob=0.5, bootstrap_frac=0.5):
    #"""
    #Randomly replace a fraction of the events with other randomly chosen events
    #Modifies in place! Use clone if you need the original tensor unmodified
    #"""
    #r = torch.rand(x.shape[0])
    #for w in range(x.shape[0]):
        #if r[w] < prob:
            #num_to_replace = int(x.shape[1] * bootstrap_frac)
            #i = torch.randperm(x.shape[1])
            #replace_idx = i[0:num_to_replace]
            #remain_idx = i[num_to_replace:]
            # Each replace_idx gets a number randomly chosen from remain_idx
            #rw = torch.randint(remain_idx.shape[0], (num_to_replace,)) #
            #x[w, replace_idx, :] = x[w, remain_idx[rw], :]
    #return x


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
    """ 
    Multiply all events in a channel by the same amount, scaling them all to be a bit bigger or smaller 
    If x has 3 dimensions, we scale each batch element separately
    If x has 2 dimensions, we scale the whole tensor
    Modifies in place! Use clone if you need the original tensor unmodified
    """
    clamp_scale = 3.0
    if len(x.shape) == 3:
        r = torch.rand(x.shape[0])
        for i in range(x.shape[0]):
            if r[i] < prob:
                z = torch.normal(mean=torch.tensor([1.0 for _ in range(x.shape[-1])]),
                        std=torch.tensor([scale for _ in range(x.shape[-1])])).to(device)
                z = torch.clamp(z, min=1.0 - clamp_scale * scale, max=1.0 + clamp_scale * scale)
                x[i, :, :] = x[i, :, :] * z
    elif len(x.shape) == 2:
        r = torch.rand(1)
        if r < prob:
            z = torch.normal(mean=torch.tensor([1.0 for _ in range(x.shape[-1])]),
                        std=torch.tensor([scale for _ in range(x.shape[-1])])).to(device)
            z = torch.clamp(z, min=1.0 - clamp_scale * scale, max=1.0 + clamp_scale * scale)
            x = x * z
    else:
        raise ValueError(f"Input tensor must have 2 or 3 dimensions (found {len(x.shape)})")
    return x


def shift(x, device='cpu', prob=0.5, scale=0.1):
    """
    Add / subtract a constant value to all events in each channel, 'shifts' all channel values up or down a bit
    If x has 3 dimensions, we shift each batch element separately
    If x has 2 dimensions, we shift the whole tensor
    Modifies in place! Use clone if you need the original tensor unmodified
    """
    clamp_scale = 3.0
    if len(x.shape) == 3:   
        r = torch.rand(x.shape[0])
        for i in range(x.shape[0]):
            if r[i] < prob:
                z = torch.normal(mean=torch.tensor([0.0 for _ in range(x.shape[-1])]), std=torch.tensor([scale for _ in range(x.shape[-1])])).to(device)
                z = torch.clamp(z, min=-clamp_scale * scale, max=clamp_scale * scale)
                x[i, :, :] = x[i, :, :] + z
    elif len(x.shape) == 2:
        r = torch.rand(1)
        if r < prob:
            z = torch.normal(mean=torch.tensor([0.0 for _ in range(x.shape[-1])]), std=torch.tensor([scale for _ in range(x.shape[-1])])).to(device)
            z = torch.clamp(z, min=-clamp_scale * scale, max=clamp_scale * scale)
            x = x + z
    else:
        raise ValueError(f"Input tensor must have 2 or 3 dimensions (found {len(x.shape)})")
    return x


def noise(x, device='cpu', prob=0.5, scale=0.1):
    """
    Adds gaussian random noise to each event, individually
    If x has 3 dimensions, we add noise to each batch element separately
    If x has 2 dimensions, we add noise to the whole tensor
    Modifies in place! Use clone if you need the original tensor unmodified
    """
    clamp_scale = 3.0
    if len(x.shape) == 3:
        r = torch.rand(x.shape[0])
        for i in range(x.shape[0]):
            if r[i] < prob:
                z = torch.normal(mean=torch.zeros_like(x[i, :, :]), std=(scale * torch.ones_like(x[i, :, :]))).to(device)
                z = torch.clamp(z, min=-clamp_scale * scale, max=clamp_scale * scale)
                x[i, :, :] = x[i, :, :] + z
    elif len(x.shape) == 2:
        r = torch.rand(1)
        if r < prob:
            z = torch.normal(mean=torch.zeros_like(x), std=(scale * torch.ones_like(x))).to(device)
            z = torch.clamp(z, min=-clamp_scale * scale, max=clamp_scale * scale)
            x = x + z
    else:
        raise ValueError(f"Input tensor must have 2 or 3 dimensions (found {len(x.shape)})")
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

class SimpleTubeSet(Dataset):

    def __init__(self, samples, transforms=None):
        self.samples = samples
        self.transforms = transforms
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        x = self.samples[i]
        if self.transforms:
            x = self.transforms(x)
        return x

    

class NoLabelTubes(Dataset):

    def __init__(self, dirpath, min_events=2048, return_key="t", transforms=None):
        self.dirpath = Path(dirpath)
        self.transforms = transforms
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
                result = item[self.return_key][idx]
            else:
                result = item[self.return_key]
            
            if self.transforms:
                result = self.transforms(result)
                
            return result
        except Exception as ex:
            raise Exception(f"Failed to load item {self.samples[i]}: {str(ex)}")
    
    def get_path(self, i):
        return self.samples[i]
        


class TubeData(Dataset):
    """
    A DataSet of flow tubes. Paths to the actual tube files (.pt) along with labels are given as an input CSV file
    This can can optionally return a subset of the standard b, t, and m tubes by using the tubes_to_return arg
    If multiple tubes are present, the return value of the __getitem__ is a dictionary keyed by tube type
    But if there's only one tube (e.g. tube_type="b"), then just the tube tensor is returned, not a dictionary
    """

    def __init__(self, labelcsv, events_to_return=-1, data_root="/", tubes_to_return=["b", "t", "m"], labelkey="label", textroot="/", report_key="text_emb",transforms=None):
        if isinstance(labelcsv, str):
            self.data = pd.read_csv(labelcsv)
        else:
            self.data = labelcsv
        self.tubes_to_return = tubes_to_return
        self.dataroot = Path(data_root)
        self.events_to_return = events_to_return
        self.labelkey = labelkey
        self.transforms = transforms

        self.report_key = report_key
        self.text_root = Path(textroot) if textroot is not None else self.dataroot

        if self.report_key is not None:
            assert self.report_key in self.data.columns, \
                f"{self.report_key} not in {self.data.columns}"
    

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        row = self.data.iloc[i]
        tubes = {}
        for tube in self.tubes_to_return:
            tubedata = torch.load(self.dataroot / row['path'], weights_only=False)
            if self.events_to_return != -1:
                if tubedata[tube].shape[0] < self.events_to_return:
                    # repeat the events in the sample until we have enough
                    num_repeats = self.events_to_return // tubedata[tube].shape[0] + 1
                    logger.info(f"Repeating {tube} {num_repeats} times to get {self.events_to_return} events")
                    tubedata[tube] = tubedata[tube].repeat(num_repeats, 1)
                    logger.info(f"New shape of {tube}: {tubedata[tube].shape}")
                    tubedata[tube] = tubedata[tube][0:self.events_to_return, :]
                    
                
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
        # If datatype of row with label is float, we just pass back the value,
        # otherwise we convert to 0/1
        if isinstance(label, float):
            label = label
        else:
            if label:
                label = 1.0
            else:
                label = 0.0

        rowdict = row.to_dict()
        rowdict['label'] = label

        # load and attach text embedding if requested 
        if self.report_key is not None:
            txt_path = row[self.report_key]
            if isinstance(txt_path, str) and len(txt_path) > 0:
                full_txt_path = self.text_root / txt_path
                text_emb = torch.load(full_txt_path, map_location="cpu")
                
                # ensure it's 1D float tensor
                if isinstance(text_emb, np.ndarray):
                    text_emb = torch.from_numpy(text_emb)
                text_emb = text_emb.float()
                rowdict["text_emb"] = text_emb
            else:
                rowdict["text_emb"] = None
        
        return tubes, rowdict
    
    def get_row_data(self, i):
        row = self.data.iloc[i]
        return row.to_dict()    

    def get_by_accession(self, accession):
        """
        Find the row in the dataframe that matches the accession, then return whatever __getitem__ does for that row
        """
        row = self.data[self.data['ACCESSION'] == accession]
        return self.__getitem__(row.index[0])
    
    def positive_negative_samples(self):
        pos = self.data[self.data[self.labelkey] == 1]
        neg = self.data[self.data[self.labelkey] == 0]
        return pos, neg


def collate_fn(items):
    """
    The NoLabelTubes dataset returns a list of tensors which may have different sizes, so we can't stack them
    """
    return items

import logging

logger = logging.getLogger(__name__)


def pil_imgloader(path):
    return Image.open(path).convert("RGB")


def tensor_loader(path):
    return torch.load(path, map_location='cpu', weights_only=False)

def numpy_loader(path):
    return np.load(path, allow_pickle=True)


class CSVDataset(Dataset):
    """
    This class reads labels and images from a CSV file. By default, it expects the CSV to have at least two
    columns: 'path' and 'label'. The 'path' column should contain the path to the image file (relative to rootdir), and the
    'label' column should contain the corresponding label. The images are loaded using PIL by default, but you can
    supply your own image loader function if you want to use a different library or method for loading images.

    """

    def __init__(
        self,
        rootdir,
        csvpath,
        reader=numpy_loader,
        label_key="label",
        path_key="path",
        label_transforms=None,
        transforms=None,
        label_first=True,
    ):
        """

        :param csvpath: CSV File with columns path and label, paths are relative to rootdir
        :param transforms: Transforms for the image (or whatever) data
        :param label_transforms: transforms for labels, if given
        :param target_transforms: transforms for labels, if not provided labels are mapped to an integer
        :param label_first: If True, returns (label, img), else return (img, label)
        """
        super().__init__()
        self.rootdir = Path(rootdir)
        self.reader = reader
        self.label_key = label_key
        self.img_key = path_key
        assert self.rootdir.is_dir(), f"{self.rootdir} is not a directory"
        self.data = pd.read_csv(csvpath)
        assert label_key in self.data.columns, f"{label_key} not in {self.data.columns}"

        if isinstance(self.img_key, str):
            assert self.img_key in self.data.columns, f"{self.img_key} not in {self.data.columns}"
        elif isinstance(self.img_key, list):
            for key in self.img_key:
                assert key in self.data.columns, f"{key} not in {self.data.columns}"
        self.transforms = transforms
        self.label_transforms = label_transforms
        self.label_first = label_first


    def __len__(self):
        return len(self.data)

    def row_info(self, row_idx):
        """Get the info of a row in the dataset, useful if your CSV has more columns than just path and label (e.g. accession)"""
        return self.data.iloc[row_idx].to_dict()
    
    def __getitem__(self, item):
        row = self.data.iloc[item]

        try:
            if isinstance(self.img_key, list):
                data = [self.reader(self.rootdir / row[key]) for key in self.img_key]
            else:
                data = self.reader(self.rootdir / row[self.img_key])
        except Exception as ex:
            logger.error(f"Could not open item {row.path}")
            raise ex

        if self.transforms:
            data = self.transforms(data)

        label = row[self.label_key]
        if self.label_transforms:
            label = self.label_transforms(label)
        
        rowinfo = row.to_dict()
        rowinfo['label'] = label
        return data, rowinfo

