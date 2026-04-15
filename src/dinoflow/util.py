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

#class WarmupCosineLRScheduler(LRScheduler):
    #""" A learning rate schedule that increases linearly from 0 to max_lr over warmup_iters, then
     #decreases using cosine decay back down to min_lr for lr_decay_iters, then stays there forever
     #"""

    #def __init__(self, optimizer, max_lr, min_lr, warmup_iters, lr_decay_iters):
        #self.optimizer = optimizer
        #self.max_lr = max_lr
        #self.min_lr = min_lr
        #self.warmup_iters = warmup_iters
        #self.lr_decay_iters = lr_decay_iters
        #self.last_lr = float("NaN")
        #super().__init__(optimizer)

    #def set_iters(self, iters):
        #self._step_count = iters

    #def get_last_lr(self):
        #return self.last_lr
    
    #def get_lr(self):
        #"""
        #Sets the LR for every param_group to be the same value
        #"""
        #warm = int(self.warmup_iters)
        #end  = int(self.lr_decay_iters)

        # 1) linear warmup from min_lr -> max_lr
        #if warm > 0 and self._step_count < warm:
            #t = (self._step_count + 1) / warm  # 0..1
            #lr = self.min_lr + t * (self.max_lr - self.min_lr)

        # 2) after decay is done, stay at min_lr
        #elif self._step_count >= end:
            #lr = self.min_lr

        # 3) cosine decay from max_lr -> min_lr
        #else:
            #denom = max(1, end - warm)  # avoid div-by-zero
            #decay_ratio = (self._step_count - warm) / denom
            #decay_ratio = float(np.clip(decay_ratio, 0.0, 1.0))
            #coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))  # 1..0
            #lr = self.min_lr + coeff * (self.max_lr - self.min_lr)

        #lrs = [float(lr) for _ in self.optimizer.param_groups]
        #self.last_lr = lrs
        #return lrs

import math
import numpy as np
from torch.optim.lr_scheduler import LRScheduler

class WarmupHoldCosineLRScheduler(LRScheduler):
    """
    Step-based schedule:
      warmup (min->max) -> hold at max -> cosine decay (max->min) -> hold at min

    Supports per-param-group max/min lrs.

    Parameters
    ----------
    warmup_iters : int
        Number of optimizer steps to warm up.
    hold_iters : int
        Number of optimizer steps to hold max LR after warmup.
    lr_decay_iters : int
        If decay_mode == "end": absolute optimizer step index where decay ends.
        If decay_mode == "steps": number of optimizer steps over which to decay.
    decay_mode : str
        "end" or "steps".
    """

    def __init__(
        self,
        optimizer,
        max_lrs,
        min_lrs,
        warmup_iters=0,
        hold_iters=0,
        lr_decay_iters=1,
        decay_mode: str = "end",
        last_epoch: int = -1,
    ):
        n_groups = len(optimizer.param_groups)

        if not isinstance(max_lrs, (list, tuple)):
            max_lrs = [max_lrs] * n_groups
        if not isinstance(min_lrs, (list, tuple)):
            min_lrs = [min_lrs] * n_groups

        self.max_lrs = [float(x) for x in max_lrs]
        self.min_lrs = [float(x) for x in min_lrs]

        assert len(self.max_lrs) == n_groups
        assert len(self.min_lrs) == n_groups

        self.warmup_iters = int(warmup_iters)
        self.hold_iters = int(hold_iters)
        self.lr_decay_iters = int(lr_decay_iters)

        if decay_mode not in ("end", "steps"):
            raise ValueError(f"decay_mode must be 'end' or 'steps', got {decay_mode}")
        self.decay_mode = decay_mode

        super().__init__(optimizer, last_epoch=last_epoch)

    def set_schedule(self, warmup_iters: int, hold_iters: int, lr_decay_iters: int, decay_mode: str | None = None):
        self.warmup_iters = int(warmup_iters)
        self.hold_iters = int(hold_iters)
        self.lr_decay_iters = int(lr_decay_iters)
        if decay_mode is not None:
            if decay_mode not in ("end", "steps"):
                raise ValueError(f"decay_mode must be 'end' or 'steps', got {decay_mode}")
            self.decay_mode = decay_mode

    def _decay_end(self, hold_end: int) -> int:
        if self.decay_mode == "end":
            # absolute end step
            return max(int(self.lr_decay_iters), hold_end)
        else:
            # duration in steps
            return hold_end + max(1, int(self.lr_decay_iters))

    def get_lr(self):
        # last_epoch increments each scheduler.step(); with Lightning interval="step",
        # last_epoch == optimizer_step_index (0,1,2,...)
        step = int(self.last_epoch)

        warm_end = max(0, int(self.warmup_iters))
        hold_end = warm_end + max(0, int(self.hold_iters))
        decay_end = self._decay_end(hold_end)

        # alpha maps [0..1] = fraction of (max-min)
        if warm_end > 0 and step < warm_end:
            # warmup: alpha from 0 -> 1 over warm_end steps
            alpha = step / float(warm_end)
            alpha = float(np.clip(alpha, 0.0, 1.0))
        elif step < hold_end:
            alpha = 1.0
        elif step >= decay_end:
            alpha = 0.0
        else:
            denom = max(1, decay_end - hold_end)
            r = (step - hold_end) / float(denom)  # 0..1
            r = float(np.clip(r, 0.0, 1.0))
            alpha = 0.5 * (1.0 + math.cos(math.pi * r))  # 1..0

        return [mn + alpha * (mx - mn) for mx, mn in zip(self.max_lrs, self.min_lrs)]

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


class FreezeEncoderWarmupCosineLRScheduler(LRScheduler):
    """ A learning rate schedule that freezes encoder/backbone components for freeze_iters,
    then increases linearly from 0 to max_lr over warmup_iters, then decreases using cosine 
    decay back down to min_lr for lr_decay_iters, then stays there forever.
    
    Encoder/backbone components are identified by parameter names containing 'encoder' or 'backbone'.
    """

    def __init__(self, optimizer, max_lr, min_lr, warmup_iters, lr_decay_iters, freeze_iters=0):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.freeze_iters = freeze_iters
        self.last_lr = float("NaN")
        
        # Identify encoder/backbone parameters
        self.encoder_param_indices = []
        self.other_param_indices = []
        
        for i, param_group in enumerate(optimizer.param_groups):
            param_name = param_group.get('name', '')
            if 'encoder' in param_name.lower() or 'backbone' in param_name.lower():
                self.encoder_param_indices.append(i)
            else:
                self.other_param_indices.append(i)
        
        super().__init__(optimizer)

    def set_iters(self, iters):
        self._step_count = iters

    def get_last_lr(self):
        return self.last_lr
    
    def get_lr(self):
        """
        Sets different LR for encoder/backbone vs other parameters
        """
        lrs = []
        # PyTorch LRScheduler increments _step_count before get_lr, so use (self._step_count - 1)
        step = self._step_count - 1
        for i, param_group in enumerate(self.optimizer.param_groups):
            if i in self.encoder_param_indices:
                # For encoder/backbone parameters
                if step < self.freeze_iters:
                    # Freeze for first freeze_iters steps
                    lr = 0.0
                else:
                    # After freeze_iters, use normal warmup cosine schedule
                    # Adjust step count to account for freeze period
                    adjusted_step = step - self.freeze_iters
                    if adjusted_step < self.warmup_iters:
                        lr = self.max_lr/2.5 * (adjusted_step) / (self.warmup_iters)
                    elif adjusted_step > self.lr_decay_iters:
                        lr = self.min_lr
                    else:
                        decay_ratio = (adjusted_step - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
                        assert 0 <= decay_ratio <= 1
                        coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
                        lr = self.min_lr + coeff * (self.max_lr/2.5 - self.min_lr)
            else:
                # For other parameters, use normal warmup cosine schedule
                if step < self.warmup_iters:
                    lr = self.max_lr * (step + 1) / (self.warmup_iters)
                elif step > self.lr_decay_iters:
                    lr = self.min_lr
                else:
                    decay_ratio = (step - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
                    assert 0 <= decay_ratio <= 1
                    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
                    lr = self.min_lr + coeff * (self.max_lr - self.min_lr)
            
            lrs.append(lr)
        
        self.last_lr = lrs
        return lrs


def _reg_target_transform_name(task_spec: dict) -> str:
    return str(task_spec.get("reg_target_transform") or "").strip().lower()


def reg_reports_physical_metrics(task_spec: dict) -> bool:
    """True if training target is non-identity and val MAE_physical / inverse maps apply."""
    return _reg_target_transform_name(task_spec) in ("arcsinh", "logit")


def reg_physical_to_training_target(y, task_spec: dict) -> float:
    """
    Map regression label from physical units (e.g. 0–100%) to training target.

    reg_target_transform:
      - (unset): identity
      - arcsinh: t = arcsinh(y / arcsinh_scale)
      - logit: t = logit(clip(y / logit_max, eps, 1-eps)) for y in [0, logit_max] (e.g. viability %)
    """
    rt = _reg_target_transform_name(task_spec)
    yf = float(y)
    if rt == "arcsinh":
        s = float(task_spec.get("arcsinh_scale", 1.0))
        if s <= 0:
            raise ValueError("arcsinh_scale must be positive when reg_target_transform=arcsinh")
        return float(np.arcsinh(yf / s))
    if rt == "logit":
        ymax = float(task_spec.get("logit_max", 100.0))
        eps = float(task_spec.get("logit_eps", 1e-4))
        if ymax <= 0:
            raise ValueError("logit_max must be positive when reg_target_transform=logit")
        if eps <= 0 or eps >= 0.5:
            raise ValueError("logit_eps must be in (0, 0.5)")
        p = yf / ymax
        p = min(max(p, eps), 1.0 - eps)
        return float(np.log(p / (1.0 - p)))
    return yf


def reg_training_to_physical(y, task_spec: dict):
    """
    Map model target back to physical units. Accepts scalar or array-like.
    """
    rt = _reg_target_transform_name(task_spec)
    ya = np.asarray(y, dtype=np.float64)
    if rt == "arcsinh":
        s = float(task_spec.get("arcsinh_scale", 1.0))
        return np.sinh(ya) * s
    if rt == "logit":
        ymax = float(task_spec.get("logit_max", 100.0))
        z = np.clip(ya, -500.0, 500.0)
        return ymax / (1.0 + np.exp(-z))
    return ya


if __name__=="__main__":
    scheduler = LinearScheduler(0.9, 1, 10)
    for i in range(20):
        print(scheduler.current_value())
        scheduler.step()
