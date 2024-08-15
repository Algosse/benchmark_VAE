import os
import itertools
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseAE
from ..base.base_utils import ModelOutput
from ...data.datasets import BaseDataset
from ...customexception import BadInheritanceError
from ..nn.base_architectures import BaseEncoder, BaseDecoder

from .vdvae_config import VDVAEConfig
from .vdvae_utils import get_1x1, get_3x3, Block, parse_layer_string, pad_channels, get_width_settings, draw_gaussian_diag_samples, gaussian_analytical_kl, DmolNet


class VDVAEEncoder(BaseEncoder):
    """
        Encoder for the VDVAE model.
        Also Prior Encoder when the model is conditioned.
        
        The condition y is expected to be a single image with the same resolution as the input data.
        So the input of the encoder is the concatenation of the input data and the condition.
        While the input od the prior encoder is only the condition.
    """

    def __init__(self, model_config: VDVAEConfig, is_prior=False):
        super().__init__()
        self.model_config = model_config
        self.is_prior = is_prior
        
        if not is_prior and model_config.is_conditioned:
            # When the model is conditioned, the encoder uses the input data and the label
            self.in_conv = get_3x3(model_config.input_dim[0] * 2, model_config.width)
        else:
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

    def forward(self, x, y=None):
        """
            Args:
                x (torch.Tensor): The input data.
                y (torch.Tensor, Optional): The condition.
        """
        
        if self.model_config.is_conditioned and not self.is_prior:
            if y is None:
                raise ValueError("The model is conditioned. The condition y is expected for the encoder.")
            else:
                x = torch.cat([x, y], dim=1)
        
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
        if model_config.is_conditioned:
            self.prior = Block(width * 2, cond_width, model_config.latent_dim * 2, residual=False, use_3x3=use_3x3, zero_last=True)
            self.resnet1 = Block(width, cond_width, width, residual=True, use_3x3=use_3x3, zero_last=True)
        else:
            self.prior = Block(width, cond_width, model_config.latent_dim * 2 + width, residual=False, use_3x3=use_3x3, zero_last=True)
        self.z_proj = get_1x1(model_config.latent_dim, width)
        self.z_proj.weight.data *= np.sqrt(1 / n_blocks)
        self.resnet2 = Block(width, cond_width, width, residual=True, use_3x3=use_3x3)
        self.resnet2.c4.weight.data *= np.sqrt(1 / n_blocks)
        self.z_fn = lambda x: self.z_proj(x)

    def sample(self, x, acts, prior_acts=None):
        qm, qv = self.enc(torch.cat([x, acts], dim=1)).chunk(2, dim=1)
        
        pm, pv, x = self.compute_prior(x, prior_acts)
        
        z = draw_gaussian_diag_samples(qm, qv)
        kl = gaussian_analytical_kl(qm, pm, qv, pv)
        return z, x, kl, dict(
            pm=pm.detach(),
            pv=pv.detach(),
            qm=qm.detach(),
            qv=qv.detach()
        )

    def sample_uncond(self, x, t=None, lvs=None):
        n, c, h, w = x.shape
        
        pm, pv, x = self.compute_prior(x)
        
        if lvs is not None:
            z = lvs
        else:
            if t is not None:
                pv = pv + torch.ones_like(pv) * np.log(t)
            z = draw_gaussian_diag_samples(pm, pv)
        return z, x, dict(
            pm=pm.detach(),
            pv=pv.detach(),
        )
    
    def compute_prior(self, x, prior_acts=None):
        if self.model_config.is_conditioned:
            pm, pv = self.prior(torch.cat([x, prior_acts], dim=1)).chunk(2, dim=1)
            x = self.resnet1(x)
        else:
            feats = self.prior(x)
            pm, pv, xpp = feats[:, :self.zdim, ...], feats[:, self.zdim:self.zdim * 2, ...], feats[:, self.zdim * 2:, ...]
            x = x + xpp
        return pm, pv, x
        

    def get_inputs(self, xs, activations, prior_activations=None):
        acts = activations[self.base]
        
        prior_acts = None
        if self.model_config.is_conditioned:
            prior_acts = prior_activations[self.base]

        try:
            x = xs[self.base]
        except KeyError:
            x = torch.zeros_like(acts)
        if acts.shape[0] != x.shape[0]:
            x = x.repeat(acts.shape[0], 1, 1, 1)
        return x, acts, prior_acts

    def forward(self, xs, activations, prior_activations=None, get_latents=False):
        x, acts, prior_acts = self.get_inputs(xs, activations, prior_activations)
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        z, x, kl, stats = self.sample(x, acts, prior_acts)
        x = x + self.z_fn(z)
        x = self.resnet2(x)
        xs[self.base] = x
        if get_latents:
            return xs, dict(z=z.detach(), kl=kl, **stats)
        return xs, dict(kl=kl)

    def forward_uncond(self, xs, t=None, lvs=None, get_latents=False):
        try:
            x = xs[self.base]
        except KeyError:
            ref = xs[list(xs.keys())[0]]
            x = torch.zeros(dtype=ref.dtype, size=(ref.shape[0], self.widths[self.base], self.base, self.base), device=ref.device)
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        z, x, stats = self.sample_uncond(x, t, lvs=lvs)
        x = x + self.z_fn(z)
        x = self.resnet2(x)
        xs[self.base] = x
        if get_latents:
            return xs, dict(z=z, **stats)
        return xs, None

class VDVAEDecoder(BaseDecoder):

    def __init__(self, model_config: VDVAEConfig):
        super().__init__()
        self.model_config = model_config
        resos = set()
        dec_blocks = []
        self.widths = get_width_settings(model_config.width, model_config.custom_width_str)
        blocks = parse_layer_string(model_config.dec_blocks)
        for idx, (res, mixin) in enumerate(blocks):
            dec_blocks.append(DecBlock(model_config, res, mixin, n_blocks=len(blocks)))
            resos.add(res)
        self.resolutions = sorted(resos)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.bias_xs = nn.ParameterList([nn.Parameter(torch.zeros(1, self.widths[res], res, res)) for res in self.resolutions if res <= model_config.no_bias_above])

        self.gain = nn.Parameter(torch.ones(1, model_config.width, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, model_config.width, 1, 1))
        self.final_fn = lambda x: x * self.gain + self.bias

    def forward(self, activations, prior_activations=None, get_latents=False):
        """
        Args:
            activations (ModelOutput): The activations from the encoder.
            prior_activations (ModelOutput, Optional): The activations from the prior encoder.
            get_latents (bool, Optional): Whether to return the latents or not.
        """
        stats = []
        xs = {a.shape[2]: a for a in self.bias_xs}
        for block in self.dec_blocks:
            xs, block_stats = block(xs, activations, prior_activations=prior_activations, get_latents=get_latents)
            stats.append(block_stats)
        xs[self.model_config.input_dim[1]] = self.final_fn(xs[self.model_config.input_dim[1]])
        
        return ModelOutput(
            recon_x=xs[self.model_config.input_dim[1]],
            stats=stats
        )

    def forward_uncond(self, n, t=None, y=None, get_latents=False):
        stats = []
        xs = {}
        for bias in self.bias_xs:
            xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)
        for idx, block in enumerate(self.dec_blocks):
            try:
                temp = t[idx]
            except TypeError:
                temp = t
            xs, block_stats = block.forward_uncond(xs, temp, get_latents=get_latents)
            stats.append(block_stats)
        xs[self.model_config.input_dim[1]] = self.final_fn(xs[self.model_config.input_dim[1]])
        
        return ModelOutput(
            recon_x=xs[self.model_config.input_dim[1]],
            stats=stats,
        )

    def forward_manual_latents(self, n, latents, t=None, get_latents=False):
        stats = []
        xs = {}
        for bias in self.bias_xs:
            xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)
        for block, lvs in itertools.zip_longest(self.dec_blocks, latents):
            xs, block_stats = block.forward_uncond(xs, t, lvs=lvs, get_latents=get_latents)
            stats.append(block_stats)
        xs[self.model_config.input_dim[1]] = self.final_fn(xs[self.model_config.input_dim[1]])

        return ModelOutput(
            recon_x=xs[self.model_config.input_dim[1]],
            stats=stats,
        )

class VDVAE(BaseAE):
    """Variational autoencoder using a very deep hierarchical architecture.
        The architecture is extended to use a prior network for conditional generation.
    
    Args:
        model_config (VDVAEConfig): The configuration of the VDVAE model.
        encoder (nn.Module): The encoder (or Bottom-up) network.
        decoder (nn.Module): The decoder (or Top-down) network.
    """
    
    def __init__(self, 
            model_config: VDVAEConfig,
            encoder: BaseEncoder,
            decoder: BaseDecoder,
            prior_encoder: Optional[BaseEncoder] = None,
            ):
        
        BaseAE.__init__(self, model_config=model_config, decoder=decoder)
        self.model_name = "VDVAE"
        
        self.model_config = model_config
        self.use_default_encoder = False
        self.set_encoder(encoder)
        
        if model_config.is_conditioned:
            self.set_prior_encoder(prior_encoder)
        else:
            prior_encoder = None
        
        if model_config.reconstruction_loss == "dmol":
            self.out_net = DmolNet(model_config) # Expect images between [-1, 1]
        elif model_config.reconstruction_loss == "mse":
            self.out_net = nn.Sequential(
                nn.Conv2d(model_config.width, model_config.input_dim[0], kernel_size=1, stride=1, padding=0),
                nn.Tanh() # resize images to [-1, 1]
            )
        else:
            raise NotImplementedError("Only dmol and mse reconstruction loss are supported for now.")
        
    def set_prior_encoder(self, prior_encoder: BaseEncoder) -> None:
        """Set the prior encoder of the model"""
        if not issubclass(type(prior_encoder), BaseEncoder):
            raise BadInheritanceError(
                (
                    "Prior Encoder must inherit from BaseEncoder class from "
                    "pythae.models.base_architectures.BaseEncoder. Refer to documentation."
                )
            )
        self.prior_encoder = prior_encoder
        
    def forward(self, inputs: BaseDataset, **kwargs):
        """Forward pass of the VDVAE model.
        
        Args:
            x (BaseDataset): The data:
                - data (torch.Tensor): The input data.
                        
        Returns:
            torch.Tensor: The reconstructed input data.
        """
        x = inputs["data"]
        
        if self.model_config.is_conditioned:
            # When the model is conditioned, the encoder uses the input data and the label while the prior only the label
            y = inputs["label"]
            enc_out = self.encoder(x, y)
            prior_out = self.prior_encoder(y)
            dec_out = self.decoder(enc_out['activations'], prior_out['activations'])
        else:
            enc_out = self.encoder(x)
            dec_out = self.decoder(enc_out['activations'])
        # dec_out.stats is a dictionnary containing the kl divergence for each block and sometime the latents
                
        sample_out = self.sample(dec_out=dec_out)
        recon_x = sample_out.recon_x
        recon_x_for_loss = sample_out.recon_x_for_loss

        loss, distortion, rate = self.loss_function(recon_x_for_loss, x, dec_out.stats)
        
        return ModelOutput(
            recon_loss=distortion,
            reg_loss=rate,
            loss=loss,
            recon_x=recon_x,
        )
                
    def loss_function(self, recon_x, x, stats):
        
        if self.model_config.reconstruction_loss == "dmol":
            distortion_per_pixel = self.out_net.nll(recon_x, x) # Reconstruction loss
        elif self.model_config.reconstruction_loss == "mse":
            distortion_per_pixel = 0.5 * F.mse_loss(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).mean(dim=-1)
        else:
            raise NotImplementedError(f"Reconstruction loss {self.model_config.reconstruction_loss} is not supported.")
        
        rate_per_pixel = torch.zeros_like(distortion_per_pixel)
        ndims = np.prod(x.shape[1:])
        for statdict in stats:
            rate_per_pixel += statdict['kl'].sum(dim=(1, 2, 3))
        rate_per_pixel /= ndims # KL divergence
        elbo = (distortion_per_pixel + rate_per_pixel).mean()
        return elbo, distortion_per_pixel.mean(), rate_per_pixel.mean()
    
    def sample(self, dec_out=None, n=6, t=None, get_latents=False):
        """Sample from the VDVAE model.
        
        Args:
            dec_out (ModelOutput): The output of the decoder. (If None, sample n unconditioned samples.)
            n (int): The number of samples to generate.
            t (torch.Tensor): The temperature of the sampling.
            
        Returns:
            torch.Tensor: The generated samples.
        """
        if not dec_out:
            dec_out = self.decoder.forward_uncond(n, t, get_latents=get_latents)
        
        if self.model_config.reconstruction_loss == "dmol":
            recon_x = self.out_net.sample(dec_out.recon_x)
            recon_x_for_loss = dec_out.recon_x
        elif self.model_config.reconstruction_loss == "mse":
            recon_x = self.out_net(dec_out.recon_x)
            recon_x_for_loss = recon_x
        else:
            raise NotImplementedError(f"Reconstruction loss {self.model_config.reconstruction_loss} is not supported.")
        
        if get_latents:
            stats = dec_out.stats
        else:
            stats = None

        return ModelOutput(
            recon_x=(recon_x + 1) / 2, # Rescale images to [0, 1]
            recon_x_for_loss=recon_x_for_loss,
            stats=stats,
        )
    
    @classmethod
    def load_from_folder(cls, dir_path):
        """Class method to be used to load the model from a specific folder

        Args:
            dir_path (str): The path where the model should have been be saved.

        .. note::
            This function requires the folder to contain:

            - | a ``model_config.json`` and a ``model.pt`` if no custom architectures were provided

            **or**

            - | a ``model_config.json``, a ``model.pt`` and a ``encoder.pkl`` (resp.
                ``decoder.pkl``) if a custom encoder (resp. decoder) was provided
        """

        """ TODO: Add the possibility to load custom encoder and decoder. Ideas:
            - Find a way to save the encoder and decoder classes
            - Load the encoder and decoder in pkl files    
        """

        model_config = VDVAEConfig.from_json_file(os.path.join(dir_path, 'model_config.json'))
        
        encoder = VDVAEEncoder(model_config)
        decoder = VDVAEDecoder(model_config)
        
        model = VDVAE(model_config, encoder, decoder)
        
        state_dict = torch.load(os.path.join(dir_path, 'model.pt'))
        model.load_state_dict(state_dict['model_state_dict'])

        return model
        
if __name__ == "__name__":
    
    RES = 28
    
    model_config = VDVAEConfig(
        input_dim=[1, RES, RES],
        reconstruction_loss="dmol",
        width=3,
        custom_width_str=None,
        enc_blocks=f"28x3,28d2,14x3,14d2,7x3",
        dec_blocks="7x3,14m7,14x3,28m14,28x3",
        bottleneck_multiple=1.0,
        no_bias_above=64,
    )
    
    encoder = VDVAEEncoder(model_config)
    decoder = VDVAEDecoder(model_config)

    model = VDVAE(model_config, encoder, decoder)
    
    x = torch.rand(10, RES, RES, 1)
    
    model(x)