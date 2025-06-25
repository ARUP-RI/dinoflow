import torch
import torch.nn as nn

class MeanPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        xt = x.transpose(1, 2) # Mean pool always operates over the last dimension, but we want to avg dim 1
        x = self.mean_pool(xt)
        x = x.squeeze(2)
        return x