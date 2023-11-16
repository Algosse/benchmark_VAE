import torch
import torch.nn as nn

from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.base_architectures import BaseDecoder

from .. import SCVAEConfig

class BasicBlock(nn.Module):
    
    def __init__(self, in_channel, out_channel, upsample=None):
        super(BasicBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        if upsample is None:
            stride = 1
            output_padding = 0
        else:
            stride = 2
            output_padding = 1
        
        self.conv1 = nn.ConvTranspose2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channel)

        self.conv2 = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, output_padding=output_padding)
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        self.relu = nn.ReLU()
        
        self.upsample = upsample
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.upsample is not None:
            residual = self.upsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out
    
class Layer(nn.Module):
    
    def __init__(self, in_channel, out_channel, nb_blocks, upsampling=True):
        super(Layer, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.nb_blocks = nb_blocks
        
        if upsampling:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size=1, stride=2, output_padding=1),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.upsample = None
        
        self.blocks = [BasicBlock(in_channel, in_channel) for _ in range(nb_blocks - 1)]
        self.blocks += [BasicBlock(in_channel, out_channel, self.upsample)]

        self.blocks = nn.Sequential(*self.blocks)
    
    def forward(self, x):
        return self.blocks(x)
    
class Decoder_ResNet_SCVAE(BaseDecoder):
    """ResNet decoder for SCVAE.
        Works with images of size 256x256.
    """
    
    def __init__(self, args: SCVAEConfig):
        super(Decoder_ResNet_SCVAE, self).__init__()
        self.output_dim = args.input_dim
        self.in_channel = self.output_dim[0]
        self.latent_dim = args.latent_dim
        self.features = args.features[::-1]
        self.nb_blocks = args.nb_blocks[::-1]
        
        self.relu = nn.ReLU()
        
        # Dense
        self.h = int(self.output_dim[1] / 2**(3 + len(self.features)))
        self.w = int(self.output_dim[2] / 2**(3 + len(self.features)))
        self.dense = nn.Linear(self.latent_dim, self.features[0] * self.h * self.w)
        
        self.upsample = nn.Upsample(scale_factor=4, mode='nearest')
        
        # Layers
        self.layers = nn.ModuleList()
        for i in range(len(self.features) - 1):
            self.layers.append(Layer(self.features[i], self.features[i+1], self.nb_blocks[i], upsampling=True))
        self.layers.append(Layer(self.features[-1], self.features[-1], self.nb_blocks[-1], upsampling=True))
        
        # Conv 1
        self.conv1 = nn.ConvTranspose2d(self.features[-1], self.in_channel, kernel_size=7, stride=2, padding=3, output_padding=1)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, z):
        
        output = ModelOutput()
        
        # Dense
        z = self.dense(z)
        z = z.view(z.size(0), -1, self.h, self.w,)
        z = self.upsample(z)
        
        # Layers        
        for i in range(len(self.layers)):
            z = self.layers[i](z)
            output["layer_{}".format(i+1)] = z
        
        z = self.conv1(z)
        z = self.bn1(z)
        z = self.sigmoid(z)
        
        output['reconstruction'] = z
        
        return output
    
    