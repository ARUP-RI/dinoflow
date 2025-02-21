import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class TubeEncoder(nn.Module):
    
    def __init__(self, num_features, model_embed_dim, layers, heads):
        super().__init__()
        self.model_dim = model_embed_dim
        self.fc = nn.Linear(num_features, model_embed_dim)
        encoderlayer = nn.TransformerEncoderLayer(d_model=model_embed_dim, nhead=heads, batch_first=True, dropout=0)
        self.encoder = nn.TransformerEncoder(encoderlayer, num_layers=layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_embed_dim))
        nn.init.xavier_normal_(self.cls_token)

    def forward(self, x):
        x = self.fc(x)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.encoder(x)
        return x[:, 0, :] # Just the cls token


class ProjectionHead(nn.Module):
    def __init__(self, model_embed_dim, hidden_dim, projection_dim):
        super().__init__()
        self.projection = nn.Linear(model_embed_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.projection2 = nn.Linear(hidden_dim, projection_dim)

    def forward(self, x):
        x = self.projection(x)
        x = self.gelu(x)
        x = self.projection2(x)
        return x

class TubeEncoderWithProjection(nn.Module):
    def __init__(self, num_features, model_embed_dim, layers, heads, hidden_dim, projection_dim):
        super().__init__()
        self.tube_encoder = TubeEncoder(num_features, model_embed_dim, layers, heads)
        self.projection_head = ProjectionHead(model_embed_dim, hidden_dim, projection_dim)

    def forward(self, x):
        x = self.tube_encoder(x)
        x = self.projection_head(x)
        return x
