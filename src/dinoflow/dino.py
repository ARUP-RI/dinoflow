#!/usr/bin/env python

import sys
from typing import List, Callable, Optional, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import yaml
from glob import glob

from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from functools import partial
import typer
import os
from pathlib import Path
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

import logging


from dinoflow.models import TubeEncoder, TubeEncoderWithProjection
from dinoflow import data
from dinoflow.loss import KoLeoLoss, CosineSimLoss, SelfCosineSimLoss, KDELoss
from dinoflow.data import scale, shift, shuffle, compose, noise, standardize_range, subsample_events, NoLabelTubes, subsample_batch

from dinoflow.util import WarmupCosineLRScheduler, random_sample, LinearScheduler

app = typer.Typer(pretty_exceptions_show_locals=False)

MASTER_PROCESS = os.environ.get('RANK', '0') == '0'
DEVICE = None # This is set in the 'train' method


logger = logging.getLogger("dinoflow")
experiment = None

logging.basicConfig(format='[%(asctime)s] %(process)d  %(name)s  %(levelname)s  %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    level=logging.INFO if MASTER_PROCESS else logging.WARNING,
                    handlers=[
                        logging.StreamHandler(),  # Output logs to stdout
                    ])

np.set_printoptions(precision=4, suppress=True, linewidth=160)

if MASTER_PROCESS:
    from comet_ml import Experiment

else:
    experiment = None



@dataclass
class TrainingConfig:
    teacher_events: int
    student_events: int
    student_reps: int
    batch_size: int
    teacher_momentum: float
    center_momentum: float
    loss_weight_sched: LinearScheduler
    lr_sched: LRScheduler
    temperature_sched: LinearScheduler


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

    S = S * S
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


def student_teacher_sample(batch, n_teacher_events, n_student_events, n_student_reps, student_augs=None, shared_augs=None, teacher_augs=None):
    """
    Sample n_teacher_events for each member of the batch, then from those events repeatedly sample n_student_events
    n_student_reps times 
    """
    assert n_student_events <= n_teacher_events
    orig_teacher_batch_size = len(batch)
    teacher_events = subsample_batch(batch, n_teacher_events) # THIS CAN CHANGE THE BATCH SIZE!! 

    # Apply shared augs here so they affect both teacher and student events
    if shared_augs is not None:
        teacher_events = shared_augs(teacher_events)

    all_student_events = []
    for i in range(n_student_reps):
        student_events = subsample_events(teacher_events, n_student_events)
        if student_augs is not None:
            student_events = student_augs(student_events.clone())

        all_student_events.append(student_events)
    

    if teacher_augs:
        teacher_events = teacher_augs(teacher_events)

    all_student_events = torch.cat(all_student_events, dim=0)
    
    assert all_student_events.shape[0] == n_student_reps * teacher_events.shape[0]
    
    return teacher_events, all_student_events


def dino_epoch(loader: DataLoader, 
               teacher: nn.Module, 
               student: nn.Module, 
               optimizer: torch.optim.Optimizer, 
               training_conf: TrainingConfig, 
               teacher_center: torch.Tensor, 
               student_augs=None, teacher_augs=None, shared_augs=None):
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
    kdeloss = KDELoss()
    # cos_sim_loss = CosineSimLoss(device=DEVICE)
    proto_cosim_loss = SelfCosineSimLoss()
    cs_loss_sum = 0
    koleo_loss_sum = 0
    proto_loss_weight = 0.0
    proto_loss_sum = 0
    report_freq = 10
    for i, batch in enumerate(loader):
        actual_batch_size = len(batch)
        n_teacher_events = int(training_conf.teacher_events)
        optimizer.zero_grad()
        logger.debug("Generating teacher and student samples")
        teacher_events, student_events = student_teacher_sample(batch, n_teacher_events, training_conf.student_events, training_conf.student_reps, student_augs=student_augs, teacher_augs=teacher_augs, shared_augs=shared_augs)
        koleo_loss_weight = training_conf.loss_weight_sched.current_value()
        actual_batch_size = teacher_events.shape[0]

        with torch.amp.autocast(enabled=enable_autocast, device_type=device_type):
            # Augment the data and do a forward pass through both the student and teacher models    
            logger.debug("Running student")
            y_s  = student(student_events.to(DEVICE).float())
            logger.debug("Running teacher")
            y_t = teacher(teacher_events.to(DEVICE).float())
            y_t = y_t.repeat(training_conf.student_reps, 1)

            logger.debug("Computing loss")
            dinoloss = dino_loss(y_s, y_t, teacher_center, s_temp=0.2, t_temp=training_conf.temperature_sched.current_value())
            koleo_batch_loss = torch.tensor(0.0).to(DEVICE)
            #Important to loop over n_student_reps here, since we don't want to include the same sample twice in the loss
            # (remember KoLeo loss looks at the two nearest neighbors, which will probably be the same sample if we include them both)
            for nr in range(training_conf.student_reps):
                # test_kde_loss = kdeloss(y_s[(nr * actual_batch_size):((nr + 1) * actual_batch_size), :])
                # test_koleo_loss = koleoloss(y_s[(nr * actual_batch_size):((nr + 1) * actual_batch_size), :])
                # logger.info(f"Test kde loss: {test_kde_loss.item()}, test koleo loss: {test_koleo_loss.item()}")
                koleo_batch_loss += kdeloss(y_s[(nr * actual_batch_size):((nr + 1) * actual_batch_size), :])
            koleo_loss = koleo_batch_loss / training_conf.student_reps
            
            protot_cosim_loss = torch.tensor(0) #proto_cosim_loss(student.module.sdpa_prototype_emb_stack.L)
            
            
            loss = dinoloss + koleo_loss_weight * koleo_loss
            koleo_loss_sum += koleo_loss.item()
            proto_loss_sum += 0 #protot_cosim_loss.item()

            epoch_loss_sum += loss.item()
            dino_loss_sum += dinoloss.item()

        logger.debug("Backprop")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        training_conf.lr_sched.step()
        training_conf.loss_weight_sched.step()
        training_conf.temperature_sched.step()
        
        param_mo = training_conf.teacher_momentum
        # Update centering and teacher weights
        if i % report_freq == 0:
            with torch.no_grad():
                cos_sim = cosine_similarity_matrix(y_t)
                cos_sim_sum += cos_sim.mean().item()

                # We are interested in how similar two sets of events sampled from the same tube are compared to two sets of events from different tubes, when 
                # run through the teacher model. 
                t2_events = subsample_batch(batch, n_teacher_events)
                assert t2_events.shape[0] == actual_batch_size, f"Whoa, didn't get batch size: {t2_events.shape}, but actual batch size: {actual_batch_size}"
                # s_events = student_events[0:actual_batch_size, :, :]
                t_events = teacher_events[0:actual_batch_size, :, :]
                y0 = teacher(t2_events.to(DEVICE).float())
                y1 = teacher(t_events.to(DEVICE).float())
                assert y0.shape[0] == y1.shape[0], f"Whoa, didn't get same output size! s_events: {t2_events.shape}, t_events: {t_events.shape}, orig student events: {student_events.shape}, orig teacher events: {teacher_events.shape}, batch: {len(batch)}"
                self_cos_sim, other_cosim = teacher_student_cosine_similarity(y0, y1, emit=MASTER_PROCESS)
                yt_self_loss_sum += self_cos_sim.item()
                yt_other_loss_sum += other_cosim.item()
    
                logger.info(f"Batch {i}, loss: {loss.item() :.4f} dino: {dinoloss.item() :.4f} cos sim: {cos_sim.mean().item() :.4f} self_cosim: {self_cos_sim.item() :.4f} other_cosim: {other_cosim.item() :.4f} teacher mo: {param_mo :.4f} teacher events: {n_teacher_events} kl weight: {koleo_loss_weight :.5f}")
                

        teacher_center = training_conf.center_momentum * teacher_center + (1 - training_conf.center_momentum) * y_t.mean(dim=0)
        dist_tot = 0
        param_tot = 0
        for param_s, param_t in zip(student.parameters(), teacher.parameters()):
            d = param_t.data - param_s.detach().data
            dist_tot += d.sum()
            param_tot += d.numel()
            param_t.data.mul_(training_conf.teacher_momentum).add_((1 - training_conf.teacher_momentum) * param_s.detach().data)


    epoch_loss = epoch_loss_sum / len(loader)
    return {
        "teacher_center": teacher_center,
        "epoch_loss": epoch_loss,
        "koleo_loss": koleo_loss_sum / len(loader),
        "cosine_sim": cos_sim_sum / len(loader),
        "self_cosim_mean": report_freq * yt_self_loss_sum / len(loader),
        "other_cosim_mean": report_freq * yt_other_loss_sum / len(loader),
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


def build_training_config(conf, optimizer):
    """
    Build a training config class from the provided config
    :param conf: Full config dictionary
    :param optimizer: Optimizer to use for training
    """
    return TrainingConfig(
        teacher_events=conf['training']['teacher_events'],
        student_events=conf['training']['n_student_events'],
        student_reps=conf['training']['n_student_reps'],
        batch_size=conf['training']['batch_size'],
        teacher_momentum=conf['training']['teacher_momentum'],
        center_momentum=conf['training']['center_momentum'],
        loss_weight_sched=LinearScheduler(conf['training']['koleo_loss_weight_start'], conf['training']['koleo_loss_weight_end'], conf['training']['koleo_loss_weight_steps']),
        lr_sched=WarmupCosineLRScheduler(optimizer, conf['training']['max_lr'], conf['training']['min_lr'], conf['training']['warmup_iters'], conf['training']['lr_decay_iters']),
        temperature_sched=LinearScheduler(conf['training']['temperature_start'], conf['training']['temperature_end'], conf['training']['temperature_steps']),
    )


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
    teacher_center = torch.zeros(conf['model']['projection_dim']).to(DEVICE)
    start_epoch = 0

    # Load from checkpoint if present
    if conf.get('checkpoint'):
        logger.info(f"Loading model from {conf['checkpoint']}")
        ckpt = torch.load(conf['checkpoint'], weights_only=False, map_location=DEVICE)
        logger.info(f"Found model configuration: {ckpt['modelconf']}")
        modelconf = ckpt['modelconf']
        student = TubeEncoderWithProjection(
            num_features=modelconf['num_features'],
            model_embed_dim=conf['model']['model_dim'],
            layers=conf['model']['layers'],
            heads=conf['model']['heads'],
            d_ff=conf['model']['d_ff'],
            hidden_dim=conf['model']['hidden_dim'],
            projection_dim=conf['model']['projection_dim']).to(DEVICE)
        
        teacher = TubeEncoderWithProjection(
            num_features=conf['model']['num_features'],
            model_embed_dim=conf['model']['model_dim'],
            layers=conf['model']['layers'],
            heads=conf['model']['heads'],
            d_ff=conf['model']['d_ff'],
            hidden_dim=conf['model']['hidden_dim'],
            projection_dim=conf['model']['projection_dim']).to(DEVICE)

        conf['model'] = modelconf
        start_epoch = conf.get("epoch", 0)
        optimizer = torch.optim.AdamW(student.parameters(), lr=conf['training']['min_lr'])
        student.load_state_dict(ckpt['student'])
        teacher.load_state_dict(ckpt['teacher'])
        optimizer.load_state_dict(ckpt['opt'])
        if ckpt.get('teacher_center') is not None:
            teacher_center = ckpt['teacher_center']
    else:
        student = TubeEncoderWithProjection(
            num_features=conf['model']['num_features'],
            model_embed_dim=conf['model']['model_dim'],
            layers=conf['model']['layers'],
            heads=conf['model']['heads'],
            d_ff=conf['model']['d_ff'],
            hidden_dim=conf['model']['hidden_dim'],
            projection_dim=conf['model']['projection_dim'],
            layer_type=conf['model']['layer_type']).to(DEVICE)
        
        teacher = TubeEncoderWithProjection(
            num_features=conf['model']['num_features'],
            model_embed_dim=conf['model']['model_dim'],
            layers=conf['model']['layers'],
            heads=conf['model']['heads'],
            d_ff=conf['model']['d_ff'],
            hidden_dim=conf['model']['hidden_dim'],
            projection_dim=conf['model']['projection_dim'],
            layer_type=conf['model']['layer_type']).to(DEVICE)

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


    training_conf = build_training_config(conf, optimizer)
    # lrschedule = WarmupCosineLRScheduler(optimizer, conf['training']['max_lr'], conf['training']['min_lr'], conf['training']['warmup_iters'], conf['training']['lr_decay_iters'])
    
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

    shared_augs = None
    #shared_augs = compose([
    #    partial(standardize_range, means=feat_means, stds=feat_stds)
    #])

    student_augs = compose([
        partial(scale, prob=0.5, scale=0.2),
        partial(shift, prob=0.45, scale=0.2),
        partial(noise, prob=0.75, scale=0.5),
    ])

    #logger.info(f"Proc: {os.getpid()} device: {device_id} w: {student.module.backbone.embedding[0].weight[0, :]}")
    for epoch in range(start_epoch, start_epoch + conf['training']['epochs']):

        epoch_results = dino_epoch(loader, teacher, student, optimizer,
                   training_conf,
                   teacher_center=teacher_center,
                   student_augs=student_augs,
                   shared_augs=shared_augs,
                   teacher_augs=None)
        teacher_center = epoch_results['teacher_center']
        cosine_sim = epoch_results['cosine_sim']
        logger.info(f"Epoch #{epoch} LR: {training_conf.lr_sched.get_lr()[0] :.5f} Loss: {epoch_results['epoch_loss'] :.4f}  cos. sim: {epoch_results['cosine_sim'] :.4f} self_cosim_mean: {epoch_results['self_cosim_mean'] :.4f} other_cosim_mean: {epoch_results['other_cosim_mean'] :.4f}")
        if experiment is not None:
            experiment.log_metric("loss", epoch_results['epoch_loss'], epoch=epoch)
            experiment.log_metric("cosine_sim", epoch_results['cosine_sim'], epoch=epoch)
            experiment.log_metric("lr", training_conf.lr_sched.get_lr()[0], epoch=epoch)
            experiment.log_metric("self_cosim_mean", epoch_results['self_cosim_mean'], epoch=epoch)
            experiment.log_metric("other_cosim_mean", epoch_results['other_cosim_mean'], epoch=epoch)
            experiment.log_metric("dino_loss", epoch_results['dino_loss'], epoch=epoch)
            experiment.log_metric("koleo_loss", epoch_results['koleo_loss'], epoch=epoch)
            experiment.log_metric("koleo_loss_weight", training_conf.loss_weight_sched.current_value(), epoch=epoch)
            experiment.log_metric("teacher_temp", training_conf.temperature_sched.current_value(), epoch=epoch)

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


def init_comet_expr():
    global experiment
    experiment = Experiment(
      api_key=os.getenv('COMET_API_KEY'),
      project_name="dinoflow",
      workspace="brendan"
    )

@app.command()
def train(config, tube_type: str = None, run_name : str = None, checkpoint: str = None):
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

    if MASTER_PROCESS and run_name is not None:
        init_comet_expr()
        experiment.set_name(run_name)
        experiment.log_parameters(conf)

    os.chdir(result_dir)
    with open("conf.yaml", "w") as fh:
        fh.write(yaml.dump(conf))

    train_dino(conf, run_name)


if __name__=="__main__":
    app()
