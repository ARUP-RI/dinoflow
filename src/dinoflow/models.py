import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)


def munge_state_dict(state_dict):
    """
    Required to load the state dict from a checkpoint into a new model
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('model.', '')
        new_state_dict[new_key] = value
    return new_state_dict


def load_btm_from_checkpoint(checkpoint: str, device=None):
    """
    Load a BTM model from the given checkpoint, doing our best to load and infer various model arch params
    :return: BTMTubes model
    """
    ckpt = torch.load(checkpoint, weights_only=False)
    if 'hyper_parameters' in ckpt:
        modelconf = ckpt['hyper_parameters']
    else:
        modelconf = {
            'd_ff': 2048,
            'model_dim': 512,
            'heads': 4,
            'layers': 10,
        }
    ckpt['state_dict'] = munge_state_dict(ckpt['state_dict'])
    if 'num_classes' not in modelconf:
        logger.info(f"num_classes not found in conf, trying to get it from model state dict..")
        bs = ckpt['state_dict']['combined.4.bias'].shape
        logger.info(f"Model final layer shape: {bs}")
        num_classes = ckpt['state_dict']['combined.4.bias'].shape[0]
    else:
        num_classes = modelconf['num_classes']
    modelconf['num_classes'] = num_classes
    logger.info(f"Model output classes: {num_classes}")
    model = BTMTubes(num_features=13,
                    model_embed_dim=modelconf['model_dim'],
                    backbone_heads=modelconf['heads'],
                    backbone_layers=modelconf['layers'],
                    d_ff=modelconf.get('d_ff', 2048),
                    output_classes=num_classes)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    return model, modelconf


def load_checkpoint(path, device=None):
    """
    Load a **backbone training** checkpoint from a file and return the tube encoder from the teacher
    """
    ckpt = torch.load(path, weights_only=False, map_location=device)
    modelconf = ckpt['modelconf']
    logger.info(f"Loading model with config: {modelconf}")    
    teacher = TubeEncoderWithProjection(
            num_features=modelconf['num_features'], 
            model_embed_dim=modelconf['model_dim'], 
            layers=modelconf['layers'],
            d_ff=modelconf.get('d_ff', 2048),
            heads=modelconf['heads'], 
            hidden_dim=modelconf['hidden_dim'], 
            projection_dim=modelconf['projection_dim'],
            layer_type=modelconf.get("layer_type", "normal")).to(device)

    teacher.load_state_dict(ckpt['teacher'])
    return teacher.tube_encoder, modelconf


def custom_activation(x):
    """ Dummy activation function that won't be used """
    return x


class SwishGLUTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0, 
                 layer_norm_eps=1e-5, batch_first=False, beta=1.0):
        # Initialize with a dummy activation that won't be used
        super().__init__(d_model=d_model, nhead=nhead, 
                         dim_feedforward=dim_feedforward,
                         dropout=dropout, layer_norm_eps=layer_norm_eps,
                         batch_first=batch_first,
                         activation=custom_activation) # Danger: Putting 'relu' or 'gelu' here will cause the _ff_block to be ignored!!
        
        # Replace the feed-forward network with SwishGLU
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear_gate = nn.Linear(d_model, dim_feedforward)
        self.beta = beta
        # Keep linear2 as the output projection
    
    def _ff_block(self, x):
        main_path = self.linear1(x)
        gate = self.linear_gate(x)
        swish = main_path * torch.sigmoid(self.beta * main_path)
        x = swish * gate
        return self.linear2(x)


class TubeEncoder(nn.Module):
    
    def __init__(self, num_features, model_embed_dim, layers, heads, d_ff=2048, layertype='normal'):
        super().__init__()
        self.model_conf = {
            'num_features': num_features,
            'model_embed_dim': model_embed_dim,
            'layers': layers,
            'heads': heads,
            'd_ff': d_ff,
            'layertype': layertype
        }
        self.model_dim = model_embed_dim
        self.num_features = num_features
        self.layers = layers
        self.heads = heads
        self.d_ff = d_ff
        assert layertype in ['normal', 'swiglu'], "Invalid layer type"
        self.fc = nn.Linear(num_features, model_embed_dim)
        if layertype == 'normal':
            encoderlayer = nn.TransformerEncoderLayer(d_model=model_embed_dim,
                                                nhead=heads,
                                                dim_feedforward=d_ff,
                                                batch_first=True,
                                                dropout=0.0)
        elif layertype == 'swiglu':
            encoderlayer = SwishGLUTransformerEncoderLayer(d_model=model_embed_dim,
                                                nhead=heads,
                                                dim_feedforward=d_ff,
                                                batch_first=True,
                                                beta=1.0,
                                                dropout=0.0)
        self.encoder = nn.TransformerEncoder(encoderlayer, num_layers=layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_embed_dim))
        nn.init.xavier_normal_(self.cls_token)

    def forward(self, x, use_cls_token=True):
        x = self.fc(x)

        if use_cls_token:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.encoder(x)
        if use_cls_token:
            return x[:, 0, :] # Just the cls token
        else:
            return x


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
    def __init__(self, num_features, model_embed_dim, layers, heads, d_ff, hidden_dim, projection_dim, layer_type='normal'):
        super().__init__()
        self.tube_encoder = TubeEncoder(num_features, model_embed_dim, layers, heads, d_ff, layer_type)
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
    def __init__(self, num_features, model_embed_dim, backbone_heads, backbone_layers, output_classes, d_ff=2048, include_classifier=True, layer_type='normal', output_scale_factor=1.0, dropout=0.1):
        super().__init__()
        self.model_conf = {
            'num_features': num_features,
            'model_embed_dim': model_embed_dim,
            'backbone_heads': backbone_heads,
            'backbone_layers': backbone_layers,
            'output_classes': output_classes,
            'd_ff': d_ff,
            'layer_type': layer_type,
            'output_scale_factor': output_scale_factor
        }
        self.output_scale_factor = output_scale_factor
        self.b_backbone = TubeEncoder(num_features, model_embed_dim, backbone_layers, backbone_heads, d_ff, layer_type)
        self.t_backbone = TubeEncoder(num_features, model_embed_dim, backbone_layers, backbone_heads, d_ff, layer_type)
        self.m_backbone = TubeEncoder(num_features, model_embed_dim, backbone_layers, backbone_heads, d_ff, layer_type)
        self.include_classifier = include_classifier
        if include_classifier:
            self.b_dropout = nn.Dropout(dropout)
            self.t_dropout = nn.Dropout(dropout)
            self.m_dropout = nn.Dropout(dropout)
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
            b_out = self.b_mlp(self.b_dropout(b_out))
            t_out = self.t_mlp(self.t_dropout(t_out))
            m_out = self.m_mlp(self.m_dropout(m_out))            
            x = self.combined(torch.cat((b_out, t_out, m_out), dim=1)) * self.output_scale_factor
        else:
            x = torch.cat((b_out, t_out, m_out), dim=1) 
        return x



class IlseBagEncoder(nn.Module):
    """
    For now the simple version where we embed the bag into a single 1D vector.
    Matrices vee, ewe, and dubya are as defined in Eq 9 of Ilse et al.
    """

    def __init__(self, embedding_dim, proto_dim, classes=1):
        """
        proto_dim is denoted L in Ilse et al, in text surrounding Eq 8 & 9.
        Roughly, L is the number of "prototype" embeddings to learn
        """
        super().__init__()
        self.vee = nn.Linear(embedding_dim, proto_dim, bias=False)
        self.ewe = nn.Linear(embedding_dim, proto_dim, bias=False)
        self.dubya = nn.Linear(proto_dim, classes, bias=False)
        self.proto_dim = proto_dim

    def forward(self, x):
        """
        Assumes input x is a bag of instances of shape
        [batch size, num inst in bag, embedding dim].
        Returns a batch of embedding vectors, i.e., a tensor of
        shape [batchsize, embedding dim]
        """

        vb = F.tanh(self.vee(x)) * F.sigmoid(self.ewe(x)) # batch size x bag size x proto_dim
        attn = self.dubya(
            vb
        )  # batch size x bag size x 1

        # next normalize the attn wts for each bag, but don't squeeze away the
        # bag size dimension until after we mult the attn weights by the bag x

        attn = F.softmax(attn, dim=-2) # attn has shape [batch, event, classes]
        #attn = F.sigmoid(attn)  # sigmoid, no dim needed

        # bag_embeddings have shape: batch size x embedding_dim
        bag_embeddings = (attn.permute((0, 2, 1)) @ x).squeeze(dim=1) # we multiple [batch, embedding_dim, event] @ [batch, event] to get [batch, embedding_dim]
        # cast up now that we have small matrices b/c cross entropy loss
        # can't handle float16
        return bag_embeddings, attn


class IlseBagModel(nn.Module):

    def __init__(self, num_features, model_embed_dim, output_classes=1, proto_dim=256, bag_classes=1):
        super().__init__()
        self.model_conf = {
            'num_features': num_features,
            'model_embed_dim': model_embed_dim,
            'output_classes': output_classes,
            'proto_dim': proto_dim,
            'bag_classes': bag_classes
        }
        self.instance_encoder = MLP(num_features, model_embed_dim, model_embed_dim, n_layers=2, residual=False)
        self.mil_attn = IlseBagEncoder(model_embed_dim, proto_dim, bag_classes)
        self.output_head = MLP(model_embed_dim * bag_classes, 128, output_classes, n_layers=2, residual=False)

    def forward(self, x):
        x = self.instance_encoder(x)
        x, attn = self.mil_attn(x)
        x = self.output_head(x.flatten(start_dim=1))
        return x
    


class IlseBagModelWithProjection(nn.Module):

    def __init__(self, num_features, model_embed_dim, output_classes=1, proto_dim=256, bag_classes=1, hidden_dim=1024, projection_dim=4096):
        super().__init__()
        self.model_conf = {
            'num_features': num_features,
            'model_embed_dim': model_embed_dim,
            'output_classes': output_classes,
            'proto_dim': proto_dim,
            'bag_classes': bag_classes
        }
        self.instance_encoder = MLP(num_features, model_embed_dim, model_embed_dim, n_layers=2, residual=False)
        self.mil_attn = IlseBagEncoder(model_embed_dim, proto_dim, bag_classes)
        # self.output_head = MLP(model_embed_dim * bag_classes, 128, output_classes, n_layers=2, residual=False)
        self.projection_head = ProjectionHead(model_embed_dim, hidden_dim, projection_dim)
        
    def forward(self, x):
        x = self.instance_encoder(x)
        x, attn = self.mil_attn(x)
        x = self.projection_head(x)
        return x
        

class Ilse3TubeModel(nn.Module):

    def __init__(self, num_features, model_embed_dim, output_classes=1, proto_dim=256, bag_classes=1):
        super().__init__()
        self.bag_output_dim = 128
        self.model_conf = {
            'num_features': num_features,
            'model_embed_dim': model_embed_dim,
            'output_classes': output_classes,
            'proto_dim': proto_dim,
            'bag_classes': bag_classes
        }
        self.m_model = IlseBagModel(num_features, model_embed_dim, self.bag_output_dim, proto_dim=proto_dim, bag_classes=bag_classes)
        self.b_model = IlseBagModel(num_features, model_embed_dim, self.bag_output_dim, proto_dim=proto_dim, bag_classes=bag_classes)
        self.t_model = IlseBagModel(num_features, model_embed_dim, self.bag_output_dim, proto_dim=proto_dim, bag_classes=bag_classes)

        self.output_head = MLP(self.bag_output_dim * 3, 128, output_classes, n_layers=2, residual=False)

    def forward(self, eventdict):
        b_events = eventdict['b'].float()
        t_events = eventdict['t'].float()
        m_events = eventdict['m'].float()

        b_out = self.b_model(b_events)
        t_out = self.t_model(t_events)
        m_out = self.m_model(m_events)

        x = torch.cat((b_out, t_out, m_out), dim=1) 
        x = self.output_head(x.flatten(start_dim=1))
        return x
    

class EncoderWithIlseMIL(nn.Module):
    def __init__(self, num_features, model_embed_dim, layers, heads, d_ff, output_classes=1, proto_dim=256, bag_classes=1):
        super().__init__()
        self.model_conf = {
            'num_features': num_features,
            'model_embed_dim': model_embed_dim,
            'layers': layers,
            'heads': heads,
            'd_ff': d_ff,
            'layertype': 'swiglu',
        }
        self.encoder = TubeEncoder(num_features, model_embed_dim, layers, heads, d_ff, layer_type='swiglu')
        self.mil_attn = IlseBagEncoder(model_embed_dim, proto_dim, bag_classes)
        self.output_head = MLP(model_embed_dim * bag_classes, 128, output_classes, n_layers=2, residual=False)
        
    def forward(self, x):
        x = self.encoder(x, use_cls_token=False)
        x, attn = self.mil_attn(x)
        x = self.output_head(x.flatten(start_dim=1))
        return x


if __name__=="__main__":
    t = EncoderWithIlseMIL(13, model_embed_dim=128, layers=4, heads=2, d_ff=2048, output_classes=1, proto_dim=256, bag_classes=1)
    x = torch.randn(32, 10, 13)
    y = t(x)
    print(y.shape)