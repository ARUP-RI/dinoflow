# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class WeightedCategoricalCrossentropy(nn.Module):
    """
    Weighted version of cross entropy

    
    cost_matrix: torch tensor with shape (num_classes, num_classes)
                    where element c[i,j] is the cost of predicting class
                    j when the true class is i.  Must be positive.
    """
    def __init__(self, cost_matrix):
        super().__init__()
        self.cost_matrix = cost_matrix

    def forward(self, y_pred, y_true):
        """
        y_pred: (N, C) where N is the batch size and C is the number of classes
        y_true: (N,) where each value is an index in [0, C-1]
        """
        num_classes = self.cost_matrix.size(0)
        N = y_pred.size(0)

        # Ensure y_true is Long tensors
        y_true = y_true.long()

        # Convert cost matrix to be indexed by y_true
        cost_vector = self.cost_matrix[y_true, :].to(y_pred.device) # (N, num_classes)

        # Standard Cross Entropy
        log_softmax = F.log_softmax(y_pred, dim=-1)
        cross_entropy = -torch.gather(log_softmax, dim=1, index=y_true.unsqueeze(1))
        cross_entropy = cross_entropy.squeeze(1)  # (N,)

        # Apply Weights
        weighted_cross_entropy = cross_entropy * torch.gather(cost_vector, dim=1, index=y_true.unsqueeze(1)).squeeze(1)

        return torch.mean(weighted_cross_entropy)


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self, device='cpu'):
        super().__init__()
        self.device=device
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'
        with torch.amp.autocast(enabled=False, device_type=device_type):
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
            I = self.pairwise_NNs_inner(student_output)  # noqa: E741
            distances = self.pdist(student_output, student_output[I])  # BxD, BxD -> B
            loss = -torch.log(distances + eps).mean()
        return loss
    


class KDELoss(nn.Module):
    """
    KDE embedding regularizer using the von Mises-Fisher (vMF) kernel. 
    Kappa is concetration paramter that controls kernel sharpness.
    """
    def __init__(self, concentration=2.0, eps=1e-8):
        super().__init__()
        self.concentration = concentration
        self.eps = eps

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (torch.Tensor): shape (B, D).
        """
        
        # Normalize embeddings to unit L2 norm
        x = F.normalize(student_output, p=2, dim=-1, eps=eps)

        # Compute cosine similarities between all pairs
        dots = torch.mm(x, x.t())  

        # Zero out diagonal self-similarities
        batch_size = x.size(0)
        dots.fill_diagonal_(0.0)

        # Apply von Mises-Fisher kernel: exp(kappa * cos(theta))
        kernel_matrix = torch.exp(self.concentration * dots)

        # Estimate density for each by summing across its neighbors
        densities = kernel_matrix.sum(dim=1) / (batch_size - 1)

        # Take negative log-density and average across batch
        loss = torch.log(densities + eps).mean()

        return loss
    

class CosineSimLoss(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

    def forward(self, ys, yt):
        """
        Args:
            ys (BxD): backbone output of student
            yt (BxD): backbone output of teacher
        """
        # Normalize each column (vector) to unit norm
        ys_norm = ys / (ys.norm(dim=0, keepdim=True) + 1e-8)  # Avoid division by zero
        yt_norm = yt / (yt.norm(dim=0, keepdim=True) + 1e-8)  # Avoid division by zero

        # Compute cosine similarity using matrix multiplication
        S = ys_norm.T @ yt_norm  # (n x m) @ (m x n) -> (n x n)

        # Elementwise product of S with itself, we do this to make all values positive, which
        # makes it so the means are small only if the magnitudes of each value are small (instead of a mix of -1s and +1s)
        S = S * S

        # On-diagonal elements represent the same sample processed through the teacher and student, so they should have high similarity
        # off diagonal elements represent different samples processed through the teacher and student, so they should have low similarity
        on_diagonal_mean = S.diagonal().mean()
        off_diagonal_mean = (S.sum() - S.diagonal().sum()) / (S.numel() - S.shape[0])
        return off_diagonal_mean - on_diagonal_mean


class SelfCosineSimLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        
        # Normalize each column (vector) to unit norm
        X_norm = x / (x.norm(dim=0, keepdim=True) + 1e-8)  # Avoid division by zero
        # Compute cosine similarity using matrix multiplication
        S = X_norm.T @ X_norm  # (n x m) @ (m x n) -> (n x n)
    
        # Compute mean of off-diagonal elements
        off_diagonal_mean = (S.sum() - S.diagonal().sum()) / (S.numel() - S.shape[0])
        return off_diagonal_mean

class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 1.0, symmetric: bool = False, detach_second: bool = True):
        super().__init__()
        self.temperature = float(temperature)
        self.symmetric = symmetric
        self.detach_second = detach_second

    def forward(self, z1, z2):
        """
        z1: [B, D] - e.g. flow projections (trainable)
        z2: [B, D] - e.g. report/text embeddings (often frozen)
        logit_scale: optional scalar tensor (exp(log_temp)) from the head.
                     If None, use 1 / self.temperature.
        """
        # Optionally freeze second branch (text / reports)
        if self.detach_second:
            with torch.no_grad():
                z2 = F.normalize(z2, dim=-1)
        else:
            z2 = F.normalize(z2, dim=-1)

        # Normalize (safe even if already normalized)
        z1 = F.normalize(z1, dim=-1)

        # Similarity matrix
        sim = z1 @ z2.T  # [B, B]

        B = z1.size(0)
        labels = torch.arange(B, device=z1.device)

        # one-way: z1 -> z2 (flow to reports)
        loss_12 = F.cross_entropy(sim, labels)

        if not self.symmetric:
            return loss_12

        # symmetric: z2 -> z1 (reports to flow)
        loss_21 = F.cross_entropy(sim.T, labels)
        
        return 0.5 * (loss_12 + loss_21)

