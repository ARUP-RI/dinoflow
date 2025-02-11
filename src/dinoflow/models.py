import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class TubeEncoder(nn.Module):
    
    def __init__(self, num_features, model_embed_dim, layers, heads):
        super().__init__()
        self.fc = nn.Linear(num_features, model_embed_dim)
        encoderlayer = nn.TransformerEncoderLayer(d_model=model_embed_dim, nhead=heads)
        self.encoder = nn.TransformerEncoder(encoderlayer, num_layers=layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_embed_dim))
        nn.init.xavier_uniform_(self.cls_token)

    def forward(self, x):
        x = self.fc(x)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.encoder(x)
        return x[:, 0, :] # Just the cls token
