import torch
import torch.nn as nn
from torch.nn import functional as F
from .utils import get_1x1, get_3x3, pad_channels

from ..hievae_config import HieVAEConfig

from pythae.models.nn.base_architectures import BaseEncoder
from pythae.models.base.base_utils import ModelOutput

class Block(nn.Module):
    
    def __init__(self, in_dim, middle_dim, out_dim, down_rate=None, residual=False, use_bn=True):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.use_bn = use_bn
        
        self.c1 = get_1x1(in_dim, middle_dim)
        self.c2 = get_3x3(middle_dim, middle_dim) 
        self.c3 = get_3x3(middle_dim, middle_dim)
        self.c4 = get_1x1(middle_dim, out_dim)
        
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(middle_dim)
            self.bn2 = nn.BatchNorm2d(middle_dim)
            self.bn3 = nn.BatchNorm2d(middle_dim)
            self.bn4 = nn.BatchNorm2d(out_dim)
            
        
    def forward(self, x):
        xhat = self.c1(F.gelu(x))
        if self.use_bn:
            xhat = self.bn1(xhat)
        xhat = self.c2(F.gelu(xhat))
        if self.use_bn:
            xhat = self.bn2(xhat)
        xhat = self.c3(F.gelu(xhat))
        if self.use_bn:
            xhat = self.bn3(xhat)
        xhat = self.c4(F.gelu(xhat))
        if self.use_bn:
            xhat = self.bn4(xhat)
        
        out = xhat + x if self.residual else xhat

        if self.down_rate is not None:
            out = F.avg_pool2d(out, kernel_size=self.down_rate, stride=self.down_rate)

        return out
    
class HVAEEncoder(BaseEncoder):
    
    def __init__(self, model_config: HieVAEConfig):
        super().__init__()
        
        
        self.model_config = model_config
        in_dim = model_config.input_dim[0]
        
        features = model_config.features if model_config.features is not None else [4,8,16,32]
        n_blocks = model_config.nb_blocks if model_config.nb_blocks is not None else [2,2,2,2]
        
        self.use_bn = model_config.use_bn
        
        self.in_conv = get_3x3(in_dim, features[0])
        
        self.layers = nn.ModuleList()
        for i in range(len(features)):
            for _ in range(n_blocks[i] - 1):
                self.layers.append(Block(features[i], features[i], features[i], down_rate=None, residual=True, use_bn=self.use_bn))
            self.layers.append(Block(features[i], features[i], features[i], down_rate=2, residual=True, use_bn=self.use_bn))
            
    def forward(self, x):
        x = self.in_conv(x)
        
        activations = {'in': x}
        
        for block in self.layers:
            w = x.shape[2]
            x = block(x)
            activations[x.shape[2]] = x

            x = x if w == x.shape[2] else pad_channels(x, x.shape[1] * 2)

        output = ModelOutput()
        output["activations"] = activations

        return output
    
if __name__ == '__main__':
    
    config = HieVAEConfig(
        input_dim=[3,64,64],
        features=[4,8,16,32],
        nb_blocks=[2,2,2,2],
    )
    
    encoder = HVAEEncoder(config)
    print(encoder)
    
    x = torch.randn(1, 3, 64, 64)
    
    out = encoder(x)['activations']
    
    for key, val in out.items():
        print(key, val.shape)