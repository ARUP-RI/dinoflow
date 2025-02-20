#!/usr/bin/env python

import sys
from typing import List

import torch
import torch.nn as nn
import yaml
from glob import glob

from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from functools import partial
import typer
import os
from pathlib import Path
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

import logging


from dinoflow.models import TubeEncoder
from dinoflow import data
from dinoflow.loss import KoLeoLoss, CosineSimLoss
from dinoflow.data import scale, shift, shuffle, compose, noise, standardize_range, subsample_events, NoLabelTubes, subsample_batch

from dinoflow.util import WarmupCosineLRScheduler, random_sample, LinearScheduler

app = typer.Typer(pretty_exceptions_show_locals=False)

MASTER_PROCESS = os.environ.get('RANK', '0') == '0'
DEVICE = None # This is set in the 'train' method


logger = logging.getLogger("dinoflow")

logging.basicConfig(format='[%(asctime)s] %(process)d  %(name)s  %(levelname)s  %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    level=logging.INFO if MASTER_PROCESS else logging.WARNING,
                    handlers=[
                        logging.StreamHandler(),  # Output logs to stdout
                    ])

np.set_printoptions(precision=4, suppress=True, linewidth=160)

if MASTER_PROCESS:
    from comet_ml import Experiment

    experiment = Experiment(
      api_key=os.getenv('COMET_API_KEY'),
      project_name="dinoflow",
      workspace="brendan"
    )
else:
    experiment = None


def cosine_similarity_matrix(X):
    # Normalize each column (vector) to unit norm
    X_norm = X / (X.norm(dim=0, keepdim=True) + 1e-8)  # Avoid division by zero
    
    # Compute cosine similarity using matrix multiplication
    S = X_norm.T @ X_norm  # (n x m) @ (m x n) -> (n x n)
    
    return S


def teacher_student_cosine_similarity(ys, yt, emit=False):
    # Normalize each column (vector) to unit norm
    ys_norm = ys / (ys.norm(dim=0, keepdim=True) + 1e-8)  # Avoid division by zero
    yt_norm = yt / (yt.norm(dim=0, keepdim=True) + 1e-8)  # Avoid division by zero
    
    # Compute cosine similarity using matrix multiplication
    S = ys_norm.T @ yt_norm  # (n x m) @ (m x n) -> (n x n)

 
    if emit:
        print(str(S[0:10, 0:10].cpu().numpy()) + "\n")

    # On-diagonal elements represent the same sample processed through the teacher and student, so they should have high similarity
    # off diagonal elements represent different samples processed through the teacher and student, so they should have low similarity
    on_diagonal_mean = S.diagonal().mean()
    off_diagonal_mean = (S.sum() - S.diagonal().sum()) / (S.numel() - S.shape[0])
    return on_diagonal_mean, off_diagonal_mean


def dino_loss(ys, yt, C, s_temp=1, t_temp=0.5, eps=1e-8):
    """ Standard Dino loss function - softmax each input vector, then take cross-entropy across them """
    sm = torch.softmax(ys / s_temp, dim=1)
    tm = torch.softmax((yt - C) / t_temp, dim=1)
    tm.detach()
    return -1 * (tm * torch.log(sm + eps)).sum(dim=1).mean()


def student_teacher_sample(batch, n_teacher_events, n_student_events, n_student_reps):
    """
    Sample n_teacher_events for each member of the batch, then from those events repeatedly sample n_student_events
    n_student_reps times 
    """
    assert n_student_events < n_teacher_events
    teacher_events = subsample_batch(batch, n_teacher_events)
    all_student_events = []
    for i in range(n_student_reps):
        student_events = subsample_events(teacher_events, n_student_events)
        all_student_events.append(student_events)
    
    teacher_events = teacher_events.repeat(n_student_reps, 1, 1)
    all_student_events = torch.cat(all_student_events, dim=0)
    return teacher_events, all_student_events


def dino_epoch(loader, teacher, student, optimizer, n_teacher_events, n_student_events, n_student_reps, center_mo, teacher_mo_schedule, teacher_center, lr_schedule, koleo_loss_weight):
    """
    Conduct a single DINO epoch
    For each batch, augment the data and pass to the student and send un-augmented data to the teacher, the loss
    tries to make them similar
    """
    enable_autocast = 'cuda' in str(DEVICE) # be careful this might break things
    scaler = torch.amp.GradScaler(enabled=enable_autocast)
    device_type = 'cuda' if 'cuda' in str(DEVICE) else 'cpu'
    
    epoch_loss_sum = 0
    cos_sim_sum = 0
    yt_self_loss_sum = 0
    yt_other_loss_sum = 0
    dino_loss_sum = 0
    koleoloss = KoLeoLoss(device=DEVICE)
    cos_sim_loss = CosineSimLoss(device=DEVICE)
    cs_loss_sum = 0
    for i, batch in enumerate(loader):
        optimizer.zero_grad()
        logger.debug("Generating teacher and student samples")
        teacher_events, student_events = student_teacher_sample(batch, n_teacher_events, n_student_events, n_student_reps)

        with torch.amp.autocast(enabled=enable_autocast, device_type=device_type):
            # Augment the data and do a forward pass through both the student and teacher models    
            logger.debug("Running student")
            y_s  = student(student_events.to(DEVICE).float())
            logger.debug("Running teacher")
            y_t = teacher(teacher_events.to(DEVICE).float())
            logger.debug("Computing loss")
            dinoloss = dino_loss(y_s, y_t, teacher_center, s_temp=1.0, t_temp=0.9)
            koleo_loss = koleoloss(y_s)
            cos_sim_loss_val = cos_sim_loss(y_s, y_t)
            cs_loss_sum += cos_sim_loss_val.item()
            loss = dinoloss + koleo_loss_weight * koleo_loss
            epoch_loss_sum += loss.item()

        logger.debug("Backprop")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if lr_schedule:
            lr_schedule.step()

        teacher_mo_schedule.step()
        param_mo = teacher_mo_schedule.current_value()
        # Update centering and teacher weights
        if i%10 == 0:
            with torch.no_grad():
                cos_sim = cosine_similarity_matrix(y_t)
                self_cos_sim, off_diag_cos_sim = teacher_student_cosine_similarity(y_s, y_t, emit=True)
                yt_self_loss_sum += self_cos_sim.item()
                yt_other_loss_sum += off_diag_cos_sim.item()
                logger.info(f"Batch {i}, loss: {loss.item() :.4f} dino: {dinoloss.item() :.4f} CS loss: {cos_sim_loss_val.item()  :.4f} cos sim: {cos_sim.mean().item() :.4f} self_cos_sim: {self_cos_sim.item() :.4f} o_d_cos_sim: {off_diag_cos_sim.item() :.4f} teacher mo: {param_mo :.4f}")
                cos_sim_sum += cos_sim.mean().item()

        teacher_center = center_mo * teacher_center + (1 - center_mo) * y_t.mean(dim=0)
        dist_tot = 0
        param_tot = 0
        for param_s, param_t in zip(student.parameters(), teacher.parameters()):
            d = param_t.data - param_s.detach().data
            dist_tot += d.sum()
            param_tot += d.numel()
            param_t.data.mul_(param_mo).add_((1 - param_mo) * param_s.detach().data)


    epoch_loss = epoch_loss_sum / len(loader)
    return {
        "teacher_center": teacher_center,
        "epoch_loss": epoch_loss,
        "cosine_sim": cos_sim_sum / len(loader),
        "yt_self_loss": yt_self_loss_sum / len(loader),
        "yt_other_loss": yt_other_loss_sum / len(loader),
        "cs_loss": cs_loss_sum / len(loader),
        "dino_loss": dino_loss_sum / len(loader)
    }


def init_ddp():
    """ Configure device for Distributed Data Parallel """
    global DEVICE
    can_use_ddp = int(os.environ.get('RANK', -1)) >= 0 and os.environ.get('WORLD_SIZE') is not None
    if not can_use_ddp:
        logger.info(f"Not using DDP, using CPU for device")
        DEVICE = 'cpu'
        return None
    else:
        logger.info(f"Using DDP, PID {os.getpid()} has rank {os.environ.get('RANK')}")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    logger.info(f"rank {rank} device count: {torch.cuda.device_count()} device id: {device_id}")
    DEVICE = f"cuda:{device_id}"
    logger.info(f"Setting cuda device to {DEVICE}")
    torch.cuda.set_device(DEVICE)
    logger.info(f"DDP [{os.getpid()}] rank: {rank} device_id: {device_id} CUDA device {DEVICE} name: {torch.cuda.get_device_name()}")
    return device_id


def train_dino(conf, run_name):
    """
    Train using the DINO self-supervised method
    :param conf:
    :param run_name:
    :return:
    """

    # When using DDP multiple processes are created, one for each GPU. Since some initialization params are random
    # (like the student weights), we need to make sure they are exactly the same across all processes
    torch.manual_seed(1785) # Important - when we initialize weights they need to be the same across all processes

    # Initialize DDP
    device_id = init_ddp()

    if 'cuda' in str(DEVICE):
        for idev in range(torch.cuda.device_count()):
            logger.info(f"CUDA device {idev} name: {torch.cuda.get_device_name({idev})}")

    if MASTER_PROCESS:
        logger.info(f"Process {os.getpid()} is the master process")

    tubes = data.NoLabelTubes(
        dirpath=conf['data']['data_dir'],
        min_events=conf['data']['input_events'] * 4,
        return_key=conf['tube_type'],
    )
    
    loader = DataLoader(tubes, batch_size=conf['training']['batch_size'], shuffle=True, pin_memory=True, num_workers=4, collate_fn=data.collate_fn)

    # Initialize here, but may be overwritten by checkpoint
    teacher_center = torch.zeros(conf['model']['model_dim']).to(DEVICE)
    start_epoch = 0

    # Load from checkpoint if present
    if conf.get('checkpoint'):
        logger.info(f"Loading model from {conf['checkpoint']}")
        ckpt = torch.load(conf['checkpoint'], weights_only=False, map_location=DEVICE)
        logger.info(f"Found model configuration: {ckpt['modelconf']}")
        modelconf = ckpt['modelconf']
        student = TubeEncoder(num_features=modelconf['num_features'], model_embed_dim=modelconf['model_dim'], layers=modelconf['layers'], heads=modelconf['heads']).to(DEVICE)
        teacher = TubeEncoder(num_features=modelconf['num_features'], model_embed_dim=modelconf['model_dim'], layers=modelconf['layers'], heads=modelconf['heads']).to(DEVICE)
        conf['model'] = modelconf
        start_epoch = conf.get("epoch", 0)
        optimizer = torch.optim.AdamW(student.parameters(), lr=conf['training']['min_lr'])
        student.load_state_dict(ckpt['student'])
        teacher.load_state_dict(ckpt['teacher'])
        optimizer.load_state_dict(ckpt['opt'])
        if ckpt.get('teacher_center') is not None:
            teacher_center = ckpt['teacher_center']
    else:
        student = TubeEncoder(num_features=conf['model']['num_features'], model_embed_dim=conf['model']['model_dim'], layers=conf['model']['layers'], heads=conf['model']['heads']).to(DEVICE)
        teacher = TubeEncoder(num_features=conf['model']['num_features'], model_embed_dim=conf['model']['model_dim'], layers=conf['model']['layers'], heads=conf['model']['heads']).to(DEVICE)
        optimizer = torch.optim.AdamW(student.parameters(), lr=conf['training']['min_lr'])

    for p in teacher.parameters():
        p.requires_grad = False

    if device_id is not None:
        student = DDP(student.to(device_id), device_ids=[device_id])
        if not conf.get('checkpoint'):
            teacher.load_state_dict(student.module.state_dict())
        # DDP throws an error if we try to use it on the teacher
        teacher = teacher.to(device_id)

        # IMPORTANT: after weight initialization set a seed thats different for every process / device_id
        torch.manual_seed(1781 * device_id + 1)
    else:
        if not conf.get('checkpoint'):
            teacher.load_state_dict(student.state_dict())


    # if 'cuda' in str(DEVICE):
    #     logger.info(f"Compiling student and teacher models")
    #     torch.compile(student)
    #     torch.compile(teacher)

    lrschedule = WarmupCosineLRScheduler(optimizer, conf['training']['max_lr'], conf['training']['min_lr'], conf['training']['warmup_iters'], conf['training']['lr_decay_iters'])
    
    model_tot_params = sum(p.numel() for p in student.parameters())
    model_trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    logger.info(f"Model total params: {model_tot_params}, trainable params: {model_trainable_params}")

    feat_means = None
    feat_stds = None
    if conf['tube_type'] == 't':
        feat_means = torch.tensor(conf['normalization_params']['t_feat_means'])
        feat_stds = torch.tensor(conf['normalization_params']['t_feat_stds'])
    elif conf['tube_type'] == 'm':
        feat_means = torch.tensor(conf['normalization_params']['m_feat_means'])
        feat_stds = torch.tensor(conf['normalization_params']['m_feat_stds'])
    elif conf['tube_type'] == 'b':
        feat_means = torch.tensor(conf['normalization_params']['b_feat_means'])
        feat_stds = torch.tensor(conf['normalization_params']['b_feat_stds'])
    else:
        raise ValueError("Unknown tube type")
    
    checkpoint_freq = conf['training']['checkpoint_freq']

    teacher_mo_schedule = LinearScheduler(conf['training']['teacher_momentum_start'], conf['training']['teacher_momentum_end'], conf['training']['teacher_momentum_steps'])

    #logger.info(f"Proc: {os.getpid()} device: {device_id} w: {student.module.backbone.embedding[0].weight[0, :]}")
    for epoch in range(start_epoch, start_epoch + conf['training']['epochs']):

        epoch_results = dino_epoch(loader, teacher, student, optimizer,
                   n_teacher_events=conf['training']['n_teacher_events'],
                   n_student_events=conf['training']['n_student_events'],
                   n_student_reps=conf['training']['n_student_reps'],
                   center_mo=conf['training']['center_momentum'],
                   teacher_mo_schedule=teacher_mo_schedule,
                   teacher_center=teacher_center,
                   lr_schedule=lrschedule,
                   koleo_loss_weight=conf['training']['koleo_loss_weight'],
                   )
        teacher_center = epoch_results['teacher_center']
        loss = epoch_results['epoch_loss']
        cosine_sim = epoch_results['cosine_sim']
        ts_self_loss = epoch_results['yt_self_loss']
        ts_other_loss = epoch_results['yt_other_loss']
        logger.info(f"Epoch #{epoch} LR: {lrschedule.get_lr()[0] :.5f} Loss: {loss :.4f}  cos. sim: {cosine_sim :.4f} yt_self_loss: {ts_self_loss :.4f} yt_other_loss: {ts_other_loss :.4f}")
        if experiment is not None:
            experiment.log_metric("loss", loss, epoch=epoch)
            experiment.log_metric("cosine_sim", cosine_sim, epoch=epoch)
            experiment.log_metric("lr", lrschedule.get_lr()[0], epoch=epoch)
            experiment.log_metric("ts_self_loss", ts_self_loss, epoch=epoch)
            experiment.log_metric("ts_other_loss", ts_other_loss, epoch=epoch)
            experiment.log_metric("dino_loss", epoch_results['dino_loss'], epoch=epoch)
        if (epoch % checkpoint_freq == 0 or epoch == (conf['training']['epochs'] - 1)) and MASTER_PROCESS:
            if isinstance(student, DDP):
                student_unwrapped = student.module
            else:
                student_unwrapped = student
            ckpt = {
                "student": student_unwrapped.state_dict(),
                "teacher": teacher.state_dict(),
                "opt": optimizer.state_dict(),
                "teacher_center": teacher_center,
                "modelconf": conf['model'],
                "trainingconf": conf['training'],
                "tube_type": conf['tube_type'],
                "feat_means": feat_means,
                "feat_stds": feat_stds,
                "epoch": epoch,
            }
            dest = f"{run_name}_epoch{epoch}.pt"
            logger.info(f"Saving checkpoint for epoch {epoch} to {dest}")
            torch.save(ckpt, dest)


@app.command()
def train(config, tube_type: str = None, run_name=None, checkpoint: str = None):
    logger.info(f"Loading config from {config}")
    conf = yaml.safe_load(open(config))
    if tube_type is not None:
        conf['tube_type'] = tube_type
    if checkpoint is not None:
        conf['checkpoint'] = checkpoint
    
    assert conf['tube_type'], f"Tube type not specified in config"

    result_root_dir = Path(conf.get("result_root", "."))
    result_root_dir.mkdir(parents=True, exist_ok=True)

    result_dir = result_root_dir / run_name
    result_dir.mkdir(parents=True, exist_ok=True)

    if experiment is not None:
        if run_name is not None:
            experiment.set_name(run_name)
        experiment.log_parameters(conf)

    os.chdir(result_dir)
    with open("conf.yaml", "w") as fh:
        fh.write(yaml.dump(conf))

    train_dino(conf, run_name)


if __name__=="__main__":
    app()
