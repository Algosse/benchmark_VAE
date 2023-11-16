import torch
import torch.nn as nn

from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.base_architectures import BaseEncoder

from .. import SCVAEConfig

class BasicBlock(nn.Module):
    
    def __init__(self, in_channel, out_channel, downsample=None):
        super(BasicBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        if downsample is None:
            stride = 1
        else:
            stride = 2
        
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        self.relu = nn.ReLU()
        
        self.downsample = downsample
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out
    
class Layer(nn.Module):
    
    def __init__(self, in_channel, out_channel, nb_blocks, downsampling=True):
        super(Layer, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.nb_blocks = nb_blocks
        
        if downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.downsample = None
        
        self.blocks = [BasicBlock(in_channel, out_channel, self.downsample)]
        self.blocks += [BasicBlock(out_channel, out_channel) for _ in range(nb_blocks - 1)]
        
        self.blocks = nn.Sequential(*self.blocks)
    
    def forward(self, x):
        return self.blocks(x)
    
class Encoder_ResNet_SCVAE(BaseEncoder):
    """ResNet encoder for SCVAE.
        Works with images of size 256x256.
    """
    
    def __init__(self, args: SCVAEConfig):
        super(Encoder_ResNet_SCVAE, self).__init__()
        self.input_dim = args.input_dim
        self.in_channel = self.input_dim[0]
        self.latent_dim = args.latent_dim
        self.features = args.features
        self.nb_blocks = args.nb_blocks
        
        # Conv 1
        self.conv1 = nn.Conv2d(self.in_channel, self.features[0], kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.features[0])
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Layers
        self.layers = nn.ModuleList()
        self.layers.append(Layer(self.features[0], self.features[0], self.nb_blocks[0], downsampling=False))
        for i in range(len(self.features) - 1):
            self.layers.append(Layer(self.features[i], self.features[i+1], self.nb_blocks[i+1], downsampling=True))
        
        # Out
        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=4)
        
        h = int(self.input_dim[1] / 2**(3 + len(self.features)))
        w = int(self.input_dim[2] / 2**(3 + len(self.features)))
        
        self.mu = nn.Linear(self.features[-1] * h * w, self.latent_dim)
        self.std = nn.Linear(self.features[-1] * h * w, self.latent_dim)

    def forward(self, x):
        
        output = ModelOutput()
        
        # Conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.max_pool(x)
        output["conv1"] = x
        # Layers        
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            output["layer_{}".format(i+1)] = x
        
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        mu = self.mu(x)
        std = self.std(x)
        
        output["embedding"] = mu
        output["log_covariance"] = std
        
        return output
    
    