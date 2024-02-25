import itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseAE
from ..base.base_utils import ModelOutput
from ...data.datasets import BaseDataset
from ..nn.base_architectures import BaseEncoder, BaseDecoder

from .vdvae_config import VDVAEConfig
from .vdvae_utils import get_1x1, get_3x3, Block, parse_layer_string, pad_channels, get_width_settings, draw_gaussian_diag_samples, gaussian_analytical_kl, DmolNet


class VDVAEEncoder(BaseEncoder):

    def __init__(self, model_config: VDVAEConfig):
        super().__init__()
        self.model_config = model_config
        
        self.in_conv = get_3x3(model_config.input_dim[0], model_config.width)
        self.widths = get_width_settings(model_config.width, model_config.custom_width_str)
        enc_blocks = []
        blockstr = parse_layer_string(model_config.enc_blocks)
        for res, down_rate in blockstr:
            use_3x3 = res > 2  # Don't use 3x3s for 1x1, 2x2 patches
            enc_blocks.append(Block(self.widths[res], int(self.widths[res] * model_config.bottleneck_multiple), self.widths[res], down_rate=down_rate, residual=True, use_3x3=use_3x3))
        n_blocks = len(blockstr)
        for b in enc_blocks:
            b.c4.weight.data *= np.sqrt(1 / n_blocks)
        self.enc_blocks = nn.ModuleList(enc_blocks)       

    def forward(self, x):
        # input shape: (batch_size, height, width, channels)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.in_conv(x)
        activations = {}
        activations[x.shape[2]] = x
        for block in self.enc_blocks:
            x = block(x)
            res = x.shape[2]
            x = x if x.shape[1] == self.widths[res] else pad_channels(x, self.widths[res])
            activations[res] = x
        
        out = ModelOutput(
            activations=activations
        )    
        
        return out


class DecBlock(nn.Module):
    def __init__(self, model_config, res, mixin, n_blocks):
        super().__init__()
        self.base = res
        self.mixin = mixin
        self.model_config = model_config
        self.widths = get_width_settings(model_config.width, model_config.custom_width_str)
        width = self.widths[res]
        use_3x3 = res > 2
        cond_width = int(width * model_config.bottleneck_multiple)
        self.zdim = model_config.latent_dim
        self.enc = Block(width * 2, cond_width, model_config.latent_dim * 2, residual=False, use_3x3=use_3x3)
        self.prior = Block(width, cond_width, model_config.latent_dim * 2 + width, residual=False, use_3x3=use_3x3, zero_last=True)
        self.z_proj = get_1x1(model_config.latent_dim, width)
        self.z_proj.weight.data *= np.sqrt(1 / n_blocks)
        self.resnet = Block(width, cond_width, width, residual=True, use_3x3=use_3x3)
        self.resnet.c4.weight.data *= np.sqrt(1 / n_blocks)
        self.z_fn = lambda x: self.z_proj(x)

    def sample(self, x, acts):
        qm, qv = self.enc(torch.cat([x, acts], dim=1)).chunk(2, dim=1)
        feats = self.prior(x)
        pm, pv, xpp = feats[:, :self.zdim, ...], feats[:, self.zdim:self.zdim * 2, ...], feats[:, self.zdim * 2:, ...]
        x = x + xpp
        z = draw_gaussian_diag_samples(qm, qv)
        kl = gaussian_analytical_kl(qm, pm, qv, pv)
        return z, x, kl

    def sample_uncond(self, x, t=None, lvs=None):
        n, c, h, w = x.shape
        feats = self.prior(x)
        pm, pv, xpp = feats[:, :self.zdim, ...], feats[:, self.zdim:self.zdim * 2, ...], feats[:, self.zdim * 2:, ...]
        x = x + xpp
        if lvs is not None:
            z = lvs
        else:
            if t is not None:
                pv = pv + torch.ones_like(pv) * np.log(t)
            z = draw_gaussian_diag_samples(pm, pv)
        return z, x

    def get_inputs(self, xs, activations):
        acts = activations[self.base]
        try:
            x = xs[self.base]
        except KeyError:
            x = torch.zeros_like(acts)
        if acts.shape[0] != x.shape[0]:
            x = x.repeat(acts.shape[0], 1, 1, 1)
        return x, acts

    def forward(self, xs, activations, get_latents=False):
        x, acts = self.get_inputs(xs, activations)
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        z, x, kl = self.sample(x, acts)
        x = x + self.z_fn(z)
        x = self.resnet(x)
        xs[self.base] = x
        if get_latents:
            return xs, dict(z=z.detach(), kl=kl)
        return xs, dict(kl=kl)

    def forward_uncond(self, xs, t=None, lvs=None):
        try:
            x = xs[self.base]
        except KeyError:
            ref = xs[list(xs.keys())[0]]
            x = torch.zeros(dtype=ref.dtype, size=(ref.shape[0], self.widths[self.base], self.base, self.base), device=ref.device)
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        z, x = self.sample_uncond(x, t, lvs=lvs)
        x = x + self.z_fn(z)
        x = self.resnet(x)
        xs[self.base] = x
        return xs

class VDVAEDecoder(BaseDecoder):

    def __init__(self, model_config: VDVAEConfig):
        super().__init__()
        self.model_config = model_config
        resos = set()
        dec_blocks = []
        self.widths = get_width_settings(model_config.width, model_config.custom_width_str)
        blocks = parse_layer_string(model_config.dec_blocks)
        print(len(blocks))
        for idx, (res, mixin) in enumerate(blocks):
            dec_blocks.append(DecBlock(model_config, res, mixin, n_blocks=len(blocks)))
            resos.add(res)
        self.resolutions = sorted(resos)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.bias_xs = nn.ParameterList([nn.Parameter(torch.zeros(1, self.widths[res], res, res)) for res in self.resolutions if res <= model_config.no_bias_above])

        self.gain = nn.Parameter(torch.ones(1, model_config.width, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, model_config.width, 1, 1))
        self.final_fn = lambda x: x * self.gain + self.bias

    def forward(self, activations, get_latents=False):
        """
        Args:
            activations (ModelOutput): The activations from the encoder.
            get_latents (bool): Whether to return the latents or not.
        """
        stats = []
        xs = {a.shape[2]: a for a in self.bias_xs}
        for block in self.dec_blocks:
            xs, block_stats = block(xs, activations, get_latents=get_latents)
            stats.append(block_stats)
        xs[self.model_config.input_dim[1]] = self.final_fn(xs[self.model_config.input_dim[1]])
        
        return ModelOutput(
            recon_x=xs[self.model_config.input_dim[1]],
            stats=stats
        )

    def forward_uncond(self, n, t=None, y=None):
        xs = {}
        for bias in self.bias_xs:
            xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)
        for idx, block in enumerate(self.dec_blocks):
            try:
                temp = t[idx]
            except TypeError:
                temp = t
            xs = block.forward_uncond(xs, temp)
        xs[self.model_config.input_dim[1]] = self.final_fn(xs[self.model_config.input_dim[1]])
        
        return ModelOutput(
            recon_x=xs[self.model_config.input_dim[1]],
            stats=None,
        )

    def forward_manual_latents(self, n, latents, t=None):
        xs = {}
        for bias in self.bias_xs:
            xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)
        for block, lvs in itertools.zip_longest(self.dec_blocks, latents):
            xs = block.forward_uncond(xs, t, lvs=lvs)
        xs[self.model_config.input_dim[1]] = self.final_fn(xs[self.model_config.input_dim[1]])
        
        return ModelOutput(
            recon_x=xs[self.model_config.input_dim[1]],
            stats=None,
        )

class VDVAE(BaseAE):
    """Variational autoencoder using a very deep hierarchical architecture.
    
    Args:
        model_config (VDVAEConfig): The configuration of the VDVAE model.
        encoder (nn.Module): The encoder (or Bottom-up) network.
        decoder (nn.Module): The decoder (or Top-down) network.
    """
    
    def __init__(self, 
            model_config: VDVAEConfig,
            encoder: BaseEncoder,
            decoder,
            ):
        
        BaseAE.__init__(self, model_config=model_config, decoder=decoder)
        self.model_name = "VDVAE"
        
        self.model_config = model_config
        self.set_encoder(encoder)
        
        if model_config.reconstruction_loss == "dmol":
            self.out_net = DmolNet(model_config) # Expect images between [-1, 1]
        else:
            self.out_net = nn.Sequential(
                nn.Conv2d(model_config.width, model_config.input_dim[0], kernel_size=1, stride=1, padding=0),
                nn.Sigmoid() # resize images to [0, 1]
            )
        
    def forward(self, inputs: BaseDataset, **kwargs):
        """Forward pass of the VDVAE model.
        
        Args:
            x (BaseDataset): The data:
                - data (torch.Tensor): The input data.
                        
        Returns:
            torch.Tensor: The reconstructed input data.
        """
        x = inputs["data"]
        
        enc_out = self.encoder(x)
        dec_out = self.decoder(enc_out['activations'])
        # dec_out.stats is a dictionnary containing the kl divergence for each block and sometime the latents
        
        if self.model_config.reconstruction_loss == "dmol":
            recon_x = self.out_net.sample(dec_out.recon_x)
        else:
            raise NotImplementedError("Only dmol reconstruction loss is supported for now.")
        
        loss, distortion, rate = self.loss_function(dec_out.recon_x, x, dec_out.stats)
        
        return ModelOutput(
            recon_loss=distortion,
            reg_loss=rate,
            loss=loss,
            recon_x=recon_x,
        )
                
    def loss_function(self, recon_x, x, stats):
        
        if self.model_config.reconstruction_loss == "dmol":
            distortion_per_pixel = self.out_net.nll(recon_x, x) # Reconstruction loss
            rate_per_pixel = torch.zeros_like(distortion_per_pixel)
            ndims = np.prod(x.shape[1:])
            for statdict in stats:
                rate_per_pixel += statdict['kl'].sum(dim=(1, 2, 3))
            rate_per_pixel /= ndims # KL divergence
            
            elbo = (distortion_per_pixel + rate_per_pixel).mean()
            return elbo, distortion_per_pixel.mean(), rate_per_pixel.mean()
        else:
            raise NotImplementedError("Only dmol reconstruction loss is supported for now.")