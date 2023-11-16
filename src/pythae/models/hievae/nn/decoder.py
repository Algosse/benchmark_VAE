import torch
import torch.nn as nn
from torch.nn import functional as F
from .utils import get_1x1, get_3x3, gaussian_analytical_kl, draw_gaussian_diag_samples

from ..hievae_config import HieVAEConfig

from pythae.models.nn.base_architectures import BaseDecoder
from pythae.models.base.base_utils import ModelOutput

class Block(nn.Module):
    
    def __init__(self, in_dim, middle_dim, out_dim, residual=False, use_bn=True):
        super().__init__()
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

        return out

class DecoderBlock(nn.Module):
    
    def __init__(self, in_dim, out_dim, z_dim, cond_prior=True, up_rate=None, use_bn=False):
        super().__init__()
        self.up_rate = up_rate
        self.z_dim = z_dim
        
        self.q = Block(in_dim*2, in_dim, z_dim*2, residual=False, use_bn=use_bn)
        if cond_prior:
            self.p = Block(in_dim * 2, in_dim, z_dim*2, residual=False, use_bn=use_bn)
        else:
            self.p = Block(in_dim, in_dim, z_dim*2, residual=False, use_bn=use_bn)
        
        self.px = Block(in_dim, in_dim, in_dim, residual=True, use_bn=use_bn)
        
        self.z_proj = get_1x1(z_dim, in_dim)
        
        self.resnet = Block(in_dim, in_dim, out_dim, residual=True, use_bn=use_bn)
        
    def forward(self, x, activations = None, prior_activations = None):
        
        if self.up_rate is not None:
            x = F.interpolate(x[:, :x.shape[1] // 2, ...], scale_factor=self.up_rate, mode='nearest')
        
        if prior_activations is not None:
            prior_activation = prior_activations[x.shape[2]]
            p = self.p(torch.cat([x, prior_activation], dim=1))
        else:
            p = self.p(x)
        pm, pv = p[:, :self.z_dim], p[:, self.z_dim:self.z_dim*2]

        x = self.px(x)
        
        if activations is not None:
            activation = activations[x.shape[2]]
            q = self.q(torch.cat([x, activation], dim=1))
            qm, qv = q[:, :self.z_dim], q[:, self.z_dim:]
            z = draw_gaussian_diag_samples(qm, qv)
            kl = gaussian_analytical_kl(qm, pm, qv, pv)
        else:
            z = draw_gaussian_diag_samples(pm, pv)
            kl = None
            
        z = self.z_proj(z)
        x = x + z
        
        x = self.resnet(x)
        
        return x, kl
    
class HVAEDecoder(BaseDecoder):
    
    def __init__(self, model_config: HieVAEConfig):
        super().__init__()
        
        self.model_config = model_config
        
        out_dim = model_config.input_dim[0]
        z_dim = model_config.latent_dim
        
        features = model_config.features[::-1] if model_config.features is not None else [32,16,8,4]
        n_blocks = model_config.nb_blocks[::-1] if model_config.nb_blocks is not None else [2,2,2,2]
        
        use_bn = model_config.use_bn
        
        cond_prior = model_config.cond_prior
        
        self.layers = nn.ModuleList()
        
        # no upsampling for the first layer
        for _ in range(n_blocks[0]):
            self.layers.append(DecoderBlock(features[0], features[0], z_dim, cond_prior=cond_prior, use_bn=use_bn))

        # upsample the rest
        for i in range(1, len(features)):
            # First block upsamples
            self.layers.append(DecoderBlock(features[i], features[i], z_dim, cond_prior=cond_prior, up_rate=2, use_bn=use_bn))
            for _ in range(n_blocks[i] - 1):
                self.layers.append(DecoderBlock(features[i], features[i], z_dim, cond_prior=cond_prior, use_bn=use_bn))
        
        
        # Works only for 64x64 images
        # TODO: compute w and h from input_dim and features
        w = model_config.input_dim[1] // 2**(len(features) - 1)
        h = model_config.input_dim[2] // 2**(len(features) - 1)
        self.x_0 = nn.Parameter(torch.zeros(1, features[0], w, h))
        
        if model_config.reconstruction_loss != 'dmol':
            if model_config.reconstruction_loss == 'normal':
                self.out_conv = get_3x3(features[-1], out_dim * 2)
            else:
                self.out_conv = get_3x3(features[-1], out_dim)
        
    def forward(self, activations = None, prior_activations = None):
        if activations is not None:
            x = self.x_0.expand(activations[self.x_0.data.shape[2]].shape[0], -1, -1, -1)
        elif prior_activations is not None:
            x = self.x_0.expand(prior_activations[self.x_0.data.shape[2]].shape[0], -1, -1, -1)
        else:
            x = self.x_0.expand(1, -1, -1, -1)
        
        kls = []
        for layer in self.layers:
            x, k = layer(x, activations, prior_activations)
            if k is not None:
                kls.append(k)

        if self.model_config.reconstruction_loss == 'dmol':
            x = x
        else:
            x = torch.sigmoid(self.out_conv(x))
        
        out = ModelOutput()
        out["recon_dist"] = x
        out["kl"] = kls
        
        return out
    
    def sample(self, n_samples):
        x = self.x_0.expand(n_samples, -1, -1, -1)
        
        for layer in self.layers:
            x, _ = layer(x, None)
        
        if self.model_config.reconstruction_loss == 'dmol':
            x = x
        else:
            x = self.out_conv(x)
        
        return x
    
if __name__ == '__main__':
    
    
    config = HieVAEConfig(
        features=[4,8,16,32],
        nb_blocks=[2,2,2,2],
        reconstruction_loss='bce',
        latent_dim=10,
        input_dim=(3,64,64),
    )
    
    decoder = HVAEDecoder(config)
    
    #print(decoder)
    
    activations = {
        8: torch.randn(1, 32, 8, 8),
        16: torch.randn(1, 16, 16, 16),
        32: torch.randn(1, 8, 32, 32),
        64: torch.randn(1, 4, 64, 64)
    }
    x, kl = decoder(activations)
    print(x.shape)
    print(kl[0].shape)