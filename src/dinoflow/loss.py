# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


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
