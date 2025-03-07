import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def load_checkpoint(path, device=None):
    """
    Load a checkpoint from a file and return the tube encoder from the teacher
    """
    ckpt = torch.load(path, weights_only=False, map_location=device)
    modelconf = ckpt['modelconf']    
    teacher = TubeEncoderWithProjection(num_features=modelconf['num_features'], model_embed_dim=modelconf['model_dim'], layers=modelconf['layers'], heads=modelconf['heads'], hidden_dim=modelconf['hidden_dim'], projection_dim=modelconf['projection_dim']).to(device)

    teacher.load_state_dict(ckpt['teacher'])
    return teacher.tube_encoder, modelconf


class TubeEncoder(nn.Module):
    
    def __init__(self, num_features, model_embed_dim, layers, heads, d_ff=2048):
        super().__init__()
        self.model_dim = model_embed_dim
        self.num_features = num_features
        self.layers = layers
        self.heads = heads
        self.d_ff = d_ff
        self.fc = nn.Linear(num_features, model_embed_dim)
        encoderlayer = nn.TransformerEncoderLayer(d_model=model_embed_dim,
                                                nhead=heads,
                                                dim_feedforward=d_ff,
                                                batch_first=True,
                                                activation='gelu',
                                                dropout=0.0)
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
    def __init__(self, num_features, model_embed_dim, layers, heads, d_ff, hidden_dim, projection_dim):
        super().__init__()
        self.tube_encoder = TubeEncoder(num_features, model_embed_dim, layers, heads, d_ff)
        self.projection_head = ProjectionHead(model_embed_dim, hidden_dim, projection_dim)

    def forward(self, x):
        x = self.tube_encoder(x)
        x = self.projection_head(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2, residual=False):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 2):
            layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)
        
        self.residual = residual
        if residual:
            assert input_dim == output_dim, "Input and output dimensions must be the same for residual connections"
        

    def forward(self, x):
        y = self.layers(x)
        if self.residual:
            y = x + y
        return y


class BTMTubes(nn.Module):
    """
    Combines the B, T, and M backbones and generates a final prediction
    """
    def __init__(self, num_features, model_embed_dim, backbone_heads, backbone_layers, output_classes, d_ff=2048, include_classifier=True):
        super().__init__()
        self.b_backbone = TubeEncoder(num_features, model_embed_dim, backbone_layers, backbone_heads, d_ff)
        self.t_backbone = TubeEncoder(num_features, model_embed_dim, backbone_layers, backbone_heads, d_ff)
        self.m_backbone = TubeEncoder(num_features, model_embed_dim, backbone_layers, backbone_heads, d_ff)
        self.include_classifier = include_classifier
        if include_classifier:
            self.b_mlp = MLP(model_embed_dim, model_embed_dim, model_embed_dim, n_layers=2, residual=True)
            self.t_mlp = MLP(model_embed_dim, model_embed_dim, model_embed_dim, n_layers=2, residual=True)
            self.m_mlp = MLP(model_embed_dim, model_embed_dim, model_embed_dim, n_layers=2, residual=True)
            self.combined = nn.Sequential(
                nn.Linear(model_embed_dim * 3, model_embed_dim),
                nn.GELU(),
                nn.Linear(model_embed_dim, model_embed_dim),
                nn.GELU(),
                nn.Linear(model_embed_dim, output_classes),
            )


    def forward(self, eventdict):
        b_events = eventdict['b'].float()
        t_events = eventdict['t'].float()
        m_events = eventdict['m'].float()

        b_out = self.b_backbone(b_events)
        t_out = self.t_backbone(t_events)
        m_out = self.m_backbone(m_events)
        if self.include_classifier:
            b_out = self.b_mlp(b_out)
            t_out = self.t_mlp(t_out)
            m_out = self.m_mlp(m_out)
        x = self.combined(torch.cat((b_out, t_out, m_out), dim=1))
        return x



