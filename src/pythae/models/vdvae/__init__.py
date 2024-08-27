"""This module is the implementation of a Very Deep VAE
(http://arxiv.org/abs/2011.10650).

Available samplers 
-------------------

TBD: Add samplers for VDVAE

.. autosummary::
    ~pythae.samplers.NormalSampler
    ~pythae.samplers.GaussianMixtureSampler
    ~pythae.samplers.TwoStageVAESampler
    ~pythae.samplers.MAFSampler
    ~pythae.samplers.IAFSampler
    :nosignatures:
"""

from .vdvae_config import VDVAEConfig
from .vdvae_model import VDVAE, VDVAEEncoder, VDVAEDecoder
from .swin_transformer_config import SwinTransformerConfig
from .swin_transformer_encoder import SwinTransformerEncoder

__all__ = ["VDVAE", "VDVAEConfig", "VDVAEEncoder", "VDVAEDecoder", "SwinTransformerConfig", "SwinTransformerEncoder"]