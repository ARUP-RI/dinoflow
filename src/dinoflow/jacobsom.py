#!/usr/bin/env python3

import argparse
import logging
import numpy as np
import os
import pandas as pd
import sys
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from functools import partial
# from time import time
import time
import pickle

DIAGNOSES = ("5-10-BNHL AML CLL Hemodilute High48Ratio LGL Myeloid N_BNHL N_Myeloid No20 "
             "NORMAL N_Quality N_TLPD PlasmaCell Reversed48Ratio").split()
TUBE_NAMES = "b t m".split()

logging.basicConfig(
    format='[%(asctime)s]  %(name)s  %(levelname)s  %(message)s',
    datefmt='%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# Strangely must import any jax related imports after logger setup or logger doesn't work
import jax
from custom_somax import SOM  # part of this flotrain package, forked from quicksom.somax
# previously used #from quicksom.somax import SOM


DEVICE_JAX = jax.devices()[0]
logger.info(f"Found jax device: {DEVICE_JAX}")
# todo replace with jax detected info
DEVICE = torch.device("cuda") if hasattr(torch, "cuda") and torch.cuda.is_available() else torch.device("cpu")
if 'cuda' in str(DEVICE):
    for idev in range(torch.cuda.device_count()):
        logger.info(f"CUDA device {idev} name: {torch.cuda.get_device_name({idev})}")


def add_log_file_handler(log_file="flotrain.log"):
    file_handler = logging.FileHandler(log_file)
    log_formatter = logging.Formatter('[%(asctime)s]  %(name)s  %(levelname)s  %(message)s', '%m-%d %H:%M:%S')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)


def project_multi(som_dir, tube_files, normed=True, num_workers=1, tube_names=TUBE_NAMES):
    """
    Make all (tubes x soms) projections needed for one case (multiple soms and multiple tube types)
    :param som_dir:
    :param tube_files:
    :param normed:
    :param num_workers:
    :param tube_names:
    :return:
    """
    projections = {}
    for tube_name, tube_file in zip(tube_names, tube_files):
        som_files = [os.path.join(som_dir, f"som_{tube_name}_{diag}.p")
                     for diag in DIAGNOSES]
        tube = np.load(tube_file)
        projections[tube_name] = [project_one(som, tube, normed=normed, num_workers=num_workers) for som in som_files]
    return projections


def project_one(som_file, tube, normed=True, num_workers=1):
    """
    Make single 2d histogram, SOM projection given SOM object and events array
    :param som_file:
    :param tube:
    :param normed:
    :param num_workers:
    :return:
    """
    start_time = time.time()
    som = SOM.load_pickle(som_file, device=DEVICE_JAX)
    bmus, _, _ = som.predict(tube, num_workers=num_workers, batch_size=10000, print_each=10000)
    # make histogram-like projection on output space
    in_dims, out_dim1, out_dim2 = som.dim, som.m, som.n
    dim1_data = bmus[:, 0]
    dim2_data = bmus[:, 1]
    projection, xi, yi = np.histogram2d(dim1_data, dim2_data, bins=(range(out_dim1 + 1), range(out_dim2 + 1)),
                                        normed=normed)
    return projection


def project_case(idx, soms=None, datasets=None, out_dir=None, normalize=None, dtype=None, tube_types=None,
                 torch_tensors=False):
    start_time = time.time()

    projections = []
    for tube in tube_types:
        # get best matching output space coordinates/units (BMUs) for events in case
        som = soms[tube]
        in_data, case = datasets[tube][idx]
        accession = case.strip().split("_")[0].split(".")[0]  # assume accession is beginning of case file name
        bmus, error, _ = som.predict(in_data, num_workers=1, batch_size=10000, print_each=10000)
        # make histogram-like projection onto output space
        in_dims, out_dim1, out_dim2 = som.dim, som.m, som.n
        dim1_data = bmus[:, 0]
        dim2_data = bmus[:, 1]
        projection, xi, yi = np.histogram2d(dim1_data, dim2_data, bins=(range(out_dim1 + 1), range(out_dim2 + 1)),
                                            normed=normalize)
        if torch_tensors:
            projection = torch.tensor(projection, dtype=dtype, device="cpu")
        projections.append(projection)

    if torch_tensors:
        out_file = f"{accession}_2d.pt"
        out_path = os.path.join(out_dir, out_file)
        torch.save(
            dict(
                tubes=projections,
                tube_order=tube_types,
            ),
            out_path
        )
    else:
        out_file = f"{accession}_2d.p"
        out_path = os.path.join(out_dir, out_file)
        pickle.dump(
            dict(
                tubes=projections,
                tube_order=tube_types,
            ),
            open(out_path, "bw")
        )


def get_tube_type(tube_type):
    if tube_type.lower() in ["b", "b_cell", "bcell"]:
        return "b_cell"
    elif tube_type.lower() in ["t", "t_cell", "tcell"]:
        return "t_cell"
    elif tube_type.lower() in ["m", "myeloid", "m_cell", "mcell"]:
        return "myeloid"
    elif not tube_type:
        return tube_type
    # this last one is a bad custom option for random tube types
    else:
        return tube_type


class TubeDataset(Dataset):
    """FlowNg tube dataset"""

    def __init__(self, csv_file=None, tensor_dir=None, transform=None, tube_type=None):  # , half_tubes=False):
        """
        Args:
            tensor_dir (string): Directory with all the tube tensors.
            transform (callable, optional): Optional transform to be applied on a sample.
            csv_file (string): Not used. Path to the csv file with annotations.
        """
        self.cases = pd.read_csv(csv_file, header=None)
        self.tensor_dir = tensor_dir
        self.transform = transform
        self.tube_type = get_tube_type(tube_type)
        if not tube_type:
            raise ValueError("tube_type required but None type given")

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, case_idx):
        if torch.is_tensor(case_idx):
            case_idx = case_idx.tolist()

        # get and check path to datafile based on data set index
        tube_path = os.path.join(self.tensor_dir,
                                 self.cases.iloc[case_idx, 0])
        if not os.path.isfile(tube_path):
            raise ValueError(f"tube file does not exist {tube_path}")

        # open and pull correct tube data from file (with lots of options to check)
        try:
            tubes = pickle.load(open(tube_path, "br"))
        except Exception as e:
            logger.debug(f"got pickle error {e}")
            logger.debug(f"tube file does not appear to be pickle format, checking next for pytorch format, {tube_path}")
            try:
                tubes = torch.load(tube_path)
            except Exception as e:
                logger.error(f"got pytorch error {e}")
                raise ValueError(f"tubes file doesn't seem to be either pickle file or pytorch file format, {tube_path}")

        if isinstance(tubes, dict) and "tube_order" not in tubes:  # assume newer, non-pytorch format
            events_data = tubes[self.tube_type]["events_data"]
            to_np = False  # don't try to change np array to np
        elif isinstance(tubes, dict) and "tube_order" in tubes:  # assume older, pytorch format
            to_np = True  # change torch array to np
            tube_order = tubes["tube_order"]
            events_data = tubes["tubes"][tube_order.index(self.tube_type)]
        else:
            raise ValueError(f"tube file is not a dict but must be")

        # transform might be type change, event limit, etc.
        if self.transform:
            events_data = self.transform(events_data, self.cases.iloc[case_idx, 0], to_np=to_np)

        return events_data


class TubeTransform(object):
    """
    makes callable objects for transform param in TubeDataset
    optionally
        - sets dtype
        - limits events
        - returns events tensor
    """

    def __init__(self, max_events=0, dtype=torch.float, add_case=False, to_np=True):
        self.max_events = max_events
        self.dtype = dtype
        self.add_case = add_case
        self.to_np = to_np

    def __call__(self, tens, case=None, to_np=None):
        if to_np is None:
            to_np = self.to_np

        if self.max_events != 0 and tens.shape[0] > self.max_events:
            transformed = tens[:self.max_events, :]  # .to(dtype=self.dtype)
        else:
            transformed = tens  # .to(dtype=self.dtype)

        if to_np:
            transformed = transformed.numpy()

        if self.add_case:
            return transformed, case
        else:
            return transformed


def custom_collate(batch):
    """
    cat all tensors along rows
    """
    batch_tensor = torch.cat(batch, 0) if isinstance(batch, list) else batch
    batch_labels = np.zeros(batch_tensor.shape[0])

    return batch_labels, batch_tensor


def custom_collate_np(batch):
    """
    cat all tensors along rows
    """
    batch_array = np.concatenate(batch, 0) if isinstance(batch, list) else batch
    batch_tensor = torch.tensor(batch_array, device="cpu")
    batch_labels = np.zeros(batch_array.shape[0])

    return batch_labels, batch_tensor


def arg_parser(argv):
    # todo convert tube data file format from torch tensor to np pickle file
    parser = argparse.ArgumentParser("Train and eval tool for SOM using quicksom package")
    subparser = parser.add_subparsers()

    # Train
    trainparser = subparser.add_parser("train", help="Train a SOM")
    trainparser.set_defaults(func=train)
    # in/out related params
    trainparser.add_argument("-d", "--tensor-dir", default=None, help="input tensor dir")
    trainparser.add_argument("-c", "--cases-file", default=None, help="input cases csv file")
    trainparser.add_argument("-o", "--out-names", nargs="+", default=['som_b.p', 'som_t.p', 'som_m.p'],
                             help="names of pickle to dump (must match tube_types count, and order)")
    trainparser.add_argument("-t", "--tube-types", type=get_tube_type, nargs="+", default=None,  # ['b', 't', 'm'],
                             help="tube type like b,t,m,B_cell,b_cell,bcell,Myeloid,etc.")
    trainparser.add_argument("-e", "--max-events", type=int, default=0,
                             help="max events to pull from each tube file (default 0, no limit)")
    trainparser.add_argument("-l", "--log", default="flotrain.log",
                               help="log file")
    # SOM related params
    trainparser.add_argument("-m", "--m", type=int, default=100, help="The width of the som")
    trainparser.add_argument("-n", "--n", type=int, default=100, help="The height of the som")
    trainparser.add_argument("--periodic", default=False, action='store_true',
                             help="if set, periodic topology is used")
    # Optimization related params
    trainparser.add_argument("--n-epoch", type=int, default=5, help="The number of iterations")
    trainparser.add_argument("-bs", "--batch-size", type=int, default=10, help="The batch size to use")
    trainparser.add_argument("--num-workers", type=int, default=0, help="The number of workers to use")
    trainparser.add_argument("--alpha", type=float, default=1.0, help="The initial learning rate")
    trainparser.add_argument("--sigma", type=float, default=None, help="The initial sigma for the convolution")
    trainparser.add_argument("--scheduler", default='linear', help="Which scheduler to use, can be linear, exp or half")
    # checkpoint and restart related params
    trainparser.add_argument("--centroids-file", default=None,help="input starting centroids file "
                             "(numpy format, .npy) typically from checkpoint, default None")
    trainparser.add_argument("--start-epoch", type=int, default=0, help="The epoch number to start at "
                             "(if restarting from checkpoint this is needed for training rate settings")
    trainparser.add_argument("--checkpoint-freq", type=int, default=1,
                             help="epoch spacing between checkpoint centroid file save, default 1")
    trainparser.add_argument("--checkpoint-name-base", default="som_checkpoint", help="name base (prefix) for "
                             "checkpoint centroid files , default som_checkpoint")

    # Project
    projectparser = subparser.add_parser("project", help="Project, predict data with existing SOM")
    projectparser.set_defaults(func=project)
    projectparser.add_argument("-d", "--tensor-dir", default=None, help="input tensor dir")
    projectparser.add_argument("-o", "--out-dir", default=None, help="output tensor dir")
    projectparser.add_argument("-c", "--cases-file", default=None, help="input cases csv file")
    projectparser.add_argument("-t", "--tube-types", type=get_tube_type, nargs="+", default=[],
                               help="tube type like b,t,m,B_cell,b_cell,bcell,Myeloid,etc. for LL panel or custom "
                               "names for other panels")
    projectparser.add_argument("-i", "--in-soms", nargs="+", default=['som_b.p', 'som_t.p', 'som_m.p'],
                               help="names of the SOM pickle files to load, must be same length and order as "
                               "'--tube-types'")
    projectparser.add_argument("--full-som-model", action="store_true", default=False,
                               help="assume --in-soms will be full som model pickle files instead of only centroid "
                               "pickle files. Centroid pickle files contain dict with keys centroids, m, and n")
    projectparser.add_argument("-nn", "--not-normalized", action="store_true", default=False,
                               help="causes the projection, 2D histogram to not be normalized to a sum of 1, "
                               "may be desired when exact event counts (not ratios) are important for a downstream "
                               "model")
    projectparser.add_argument("-e", "--max-events", type=int, default=0,
                               help="max events to pull from each tube file (default 0, no limit)")
    projectparser.add_argument("--num-workers", type=int, default=0,
                               help="The number of workers to use")
    projectparser.add_argument("-s", "--chunk-size", type=int, default=0,
                               help="The cases chunk size for processing a chunk of cases. If chunk is "
                               "smaller than chunk size, existing chunk cases do get processed. (Default "
                               "= 0, process all cases)")
    projectparser.add_argument("-ci", "--chunk-index", type=int, default=None,
                               help="The cases chunk index for processing a chunk of cases. If index > "
                               "existing chunks, zero cases process and no error raised.")
    projectparser.add_argument("-l", "--log", default="flotrain.log",
                               help="log file")

    # Project One Case
    project1parser = subparser.add_parser("project1case", help="Project1case, project 1 case with existing SOM(s) for "
                                          "one or more tube types")
    project1parser.set_defaults(func=project1case)
    project1parser.add_argument("-i", "--input-tubes", nargs="+", required=True,
                                help="paths to input tube event files, one path per tube in tube type order "
                                     "'b', 't', 'm'")
    project1parser.add_argument("-o", "--output_features", required=True,
                                help="output feature vector for case, numpy .npy binary format")
    project1parser.add_argument("--num-workers", type=int, default=1,
                                help="The number of workers to use")


    args = parser.parse_args(argv)

    # run checks
    if args.func == train and len(args.tube_types) != len(args.out_names) and args.tube_types :
        raise ValueError("number of tube_types to train must equal number of out_names for trained models")

    if vars(args) and hasattr(args, 'func'):
        return args
    print(parser.print_help())
    return None


def train(
        # In/Out
        centroids_file=None,
        checkpoint_freq=1,
        checkpoint_name_base="som_checkpoint",
        tensor_dir=None,
        cases_file=None,
        out_names=None,
        tube_types=("b_cell",),
        max_events=300000,
        dtype=torch.float,
        # SOM
        m=32,
        n=32,
        # periodic=False,
        # Optim
        n_epoch=5,
        start_epoch=0,
        # chunk_size=200000,
        batch_size=1,
        num_workers=0,
        alpha=None,
        sigma=None,
        sched="linear",
        log="flotrain.log",
        **kwargs,
):
    """
    :param centroids_file:
    :param checkpoint_freq:
    :param checkpoint_name_base:
    :param n_epoch:
    :param start_epoch:
    :param tensor_dir:
    :param cases_file:
    :param out_names:
    :param tube_types:
    :param max_events:
    :param dtype:
    :param m:
    :param n:
    # :param periodic:
    # :param chunk_size:
    :param batch_size:
    :param num_workers:
    :param alpha:
    :param sigma:
    :param sched:
    :param log:
    # :param cases:
    :param kwargs:
    :return:
    """
    if not tube_types:
        tube_types = [None] * len(out_names)

    for tube_type, out_name in zip(tube_types, out_names):
        tube_transform = TubeTransform(max_events=max_events, dtype=dtype)
        dataset = TubeDataset(tensor_dir=tensor_dir, csv_file=cases_file, tube_type=tube_type, transform=tube_transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
                            collate_fn=custom_collate_np)
        input_dim = dataset[0].shape[1]

        # start from previous trained SOM centroids or from scratch
        if centroids_file:  # starting from previous trained SOM
            centroids = jax.device_put(np.load(centroids_file))
        else:  # starting from scratch
            centroids = None

        # initialize, train SOM object
        som = SOM(m=m, n=n, dim=input_dim, centroids=centroids, alpha=alpha, sigma=sigma, sched=sched,
                  device=DEVICE_JAX)
        learning_error = som.fit(loader, n_epoch=n_epoch, start_epoch=start_epoch, batch_size=batch_size,
                                 checkpoint_freq=checkpoint_freq, checkpoint_name_base=checkpoint_name_base,
                                 num_workers=num_workers,)
        # todo remove line? Apparently not needed?
        # som.to_device(jax.devices("cpu")[0])
        # np.save(out_name, som.centroids)
        centroids_dict = dict(centroids=som.centroids, m=m, n=n)
        pickle.dump(centroids_dict, open(out_name, "bw"))



def project(
        tensor_dir=None,
        cases_file=None,
        tube_types=None,
        in_soms=None,
        full_som_model=False,
        out_dir=None,
        max_events=1000,
        dtype=np.float,
        not_normalized=False,
        chunk_size=500,
        chunk_index=None,
        log="flotrain.log",
        **kwargs,
):
    """
    todo stop using torch, pickle projection file with 3 tubes in np format
    :param tensor_dir:
    :param cases_file:
    :param tube_types: example: ("b_cell",)
    :param in_soms: example: ("som_b.p",)
    :param full_som_model:
    :param out_dir:
    :param max_events:
    :param dtype: examples: np.float, torch.float
    :param normalize:
    :param chunk_size:
    :param chunk_index:
    :param log:
    :param kwargs:
    :return:
    """
    # make sure ouput dir exists
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # get cases list for logging
    cases_list = [name.strip() for name in open(cases_file).readlines()]

    # build SOMs and DataSets for projections
    tube_transform = TubeTransform(max_events=max_events, dtype=dtype, add_case=True)
    soms = {}
    datasets = {}
    for som, tube in zip(in_soms, tube_types):
        if not full_som_model:
            som_dict = pickle.load(open(som, "br"))
            centroids = som_dict["centroids"]
            input_dim = centroids.shape[1]
            m, n = som_dict["m"], som_dict["n"]
            tubes_som = SOM(m=m, n=n, dim=input_dim, centroids=centroids, device=DEVICE_JAX)
        else:
            tubes_som = SOM.load_pickle(inname=som, device=DEVICE_JAX)
        soms[tube] = tubes_som
        datasets[tube] = TubeDataset(tensor_dir=tensor_dir, csv_file=cases_file, tube_type=tube,
                                     transform=tube_transform)  # , half_tubes=False)

    # find chunk of cases to project
    if not chunk_size:
        start_index = 0
        end_index = len(datasets[tube_types[0]])
        logger.info(f"will project full list of {end_index} case files from '{cases_file}'")
    else:
        start_index = chunk_size * chunk_index
        end_index = chunk_size * (chunk_index + 1)
        cases_file_length = len(datasets[tube_types[0]])
        if cases_file_length <= start_index:
            logger.info(
                f"no projections will be generated\n" 
                f"because cases_file start_index {start_index} (chunk_size {chunk_size} * chunk_index {chunk_index})\n"
                f"is beyond the full length {cases_file_length} of the cases file {cases_file} ")
            end_index = start_index
        elif cases_file_length < end_index:
            logger.info(
                f"only partial chunk of projections will be generated\n"
                f"because cases_file end_index {end_index} (chunk_size {chunk_size} * chunk_index+1 {chunk_index+1})\n"
                f"is beyond the full length {cases_file_length} of the cases file {cases_file} ")
            end_index = cases_file_length
        else:
            logger.info(
                f"full chunk of projections will be generated\n"
                f"from cases_file start_index {start_index} (chunk_size {chunk_size} * chunk_index {chunk_index})\n"
                f"to cases_file end_index {end_index} (chunk_size {chunk_size} * chunk_index+1 {chunk_index+1})\n"
                f"from the total {cases_file_length} cases in the cases_file {cases_file} ")

    # projections
    # (written as a partial function in case we wanted to multiprocess (currently causes jax issues))
    project_case_by_index = partial(project_case, soms=soms, datasets=datasets, out_dir=out_dir,
                                    normalize=not not_normalized, dtype=dtype, tube_types=tube_types)

    start_time = time.time()
    for idx in range(start_index, end_index):
        case_name = cases_list[idx]
        case_start_time = time.time()
        project_case_by_index(idx)
        case_end_time = time.time()
        case_time = case_end_time - case_start_time  # in sec
        running_time = (case_end_time - start_time) / 60.0  # in minutes
        cases_done = idx - start_index + 1
        cases_left = end_index - idx
        eta = running_time / cases_done * cases_left  # in minutes
        logger.info(f"case {idx + 1} of {end_index-start_index}, {case_name}, case_time={case_time:.1f}sec, "
                    f"run_time={running_time:.1f}min, eta={eta:.1f}min")

    wall_time = time.time() - start_time
    logger.info(f"Finished projection of {end_index - start_index} tensor files in {wall_time:.4f} seconds.")


def project1case(
        input_tubes=None,
        som_dir=None,
        output_features=None,
        num_workers=1,
        normalize=True,
        tube_names=("b", "t", "m"),
):

    # make 45 projections = 15 different diagnosis SOM projections x 3 tube types
    projections = project_multi(som_dir, input_tubes, normed=normalize, num_workers=num_workers, tube_names=tube_names)
    # order projections to match training order of gradient boost model
    ordered = []
    for diag in DIAGNOSES:
        for tube_name in tube_names:
            ordered.append(projections[tube_name][diag])
    # finally, vectorize combined projections
    features = np.concatenate(ordered).flatten()
    np.save(output_features, features)


def main(argv):
    args = arg_parser(argv)
    if args:
        # add log file
        add_log_file_handler(log_file=args.log)
        # run
        args.func(**vars(args))


if __name__ == "__main__":
    main(sys.argv[1:])
