import logging

import torch
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import LRScheduler

import yaml


logger = logging.getLogger(__name__)


def load_conf(conf_file, **kwargs):
    with open(conf_file) as fh:
        conf = yaml.safe_load(fh)
    conf.update((k,v) for k,v in kwargs.items() if v is not None)
    return conf


def random_sample(x, max_size=5000):
    """ Randomly select the first max_size rows (elements of first dimension) to return """
    if x.shape[0] <= max_size:
        return x
    idx = torch.randperm(x.shape[0])[0:max_size]
    return x[idx, :]


def lr_for_iter(step_count, warmup_iters, max_lr, min_lr, lr_decay_iters):
    if step_count < warmup_iters:
        lr = max_lr * (step_count + 1) / (warmup_iters)
    elif step_count > lr_decay_iters:
        lr = min_lr
    else:
        decay_ratio = (step_count - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))  # coeff ranges 0..1
        lr = min_lr + coeff * (max_lr - min_lr)
    return lr


def lr_for_iter(step_count, warmup_iters, max_lr, min_lr, lr_decay_iters):
    if step_count < warmup_iters:
        lr = max_lr * (step_count + 1) / (warmup_iters)
    elif step_count > lr_decay_iters:
        lr = min_lr
    else:
        decay_ratio = (step_count - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))  # coeff ranges 0..1
        lr = min_lr + coeff * (max_lr - min_lr)
    return lr


class ConstrantLRScheduler(LRScheduler):
    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        self.lr = lr
        super().__init__(optimizer)

    def get_lr(self):
        return [self.lr for _ in self.optimizer.param_groups]

class WarmupCosineLRScheduler(LRScheduler):
    """ A learning rate schedule that increases linearly from 0 to max_lr over warmup_iters, then
     decreases using cosine decay back down to min_lr for lr_decay_iters, then stays there forever
     """

    def __init__(self, optimizer, max_lr, min_lr, warmup_iters, lr_decay_iters):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.last_lr = float("NaN")
        super().__init__(optimizer)

    def set_iters(self, iters):
        self._step_count = iters

    def get_last_lr(self):
        return self.last_lr
    
    def get_lr(self):
        """
        Sets the LR for every param_group to be the same value
        """
        # 1) linear warmup for warmup_iters steps
        if self._step_count < self.warmup_iters:
            lr = self.max_lr * (self._step_count + 1) / (self.warmup_iters)
        # 2) if it > lr_decay_iters, return min learning rate
        elif self._step_count > self.lr_decay_iters:
            lr = self.min_lr
        else:
           # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (self._step_count - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))  # coeff ranges 0..1
            lr = self.min_lr + coeff * (self.max_lr - self.min_lr)
        lr = [lr for _ in self.optimizer.param_groups]
        self.last_lr = lr
        return lr


class LinearScheduler():

    def __init__(self, start_value, end_value, num_steps):
        self.start_value = start_value
        self.end_value = end_value
        self.num_steps = num_steps
        self.current_step = 0

    def current_value(self):
        return self.value_for_step(self.current_step)

    def step(self):
        self.current_step += 1

    def value_for_step(self, step):
        if step > self.num_steps:
            return self.end_value
        return self.start_value + step * (self.end_value - self.start_value) / self.num_steps


def munge_label_df(labelcsv):
    bertp = pd.read_csv(labelcsv, dtype={"accession": str})
    bertp['Any_BNHL'] = bertp[['10+BNHL', '5+BNHL', '5-10-BNHL', 'B-NHL', 'BALL']].any(axis=1)
    bertp['viability'].fillna(-1, inplace=True)
    bertp['diagnoses'] = bertp['diagnoses'].astype(str, copy=True)
    return bertp

if __name__=="__main__":
    scheduler = LinearScheduler(0.9, 1, 10)
    for i in range(20):
        print(scheduler.current_value())
        scheduler.step()
