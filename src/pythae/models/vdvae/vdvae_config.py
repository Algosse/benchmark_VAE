from typing import Literal, Optional

from ..vae.vae_config import VAEConfig

from pydantic.dataclasses import dataclass

@dataclass
class VDVAEConfig(VAEConfig):
    """VDVAE config class.

    Modified version of https://github.com/openai/vdvae/blob/main/hps.py

    Parameters:
        input_dim (tuple): The input_data dimension. (C x H x W)
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        width (int): The width of the first convolutional layer.
        custom_width_str (str): A string to specify custom width for each resolution. If None, width is constant.
        enc_blocks (str): A string to specify the architecture of the encoder.
        dec_blocks (str): A string to specify the architecture of the decoder.
        bottleneck_multiple (float): The width of the middle layer in the blocks is this times the input width.
        no_bias_above (int): The resolution above which to not use bias in convolutions.
        num_mixtures (int): The number of mixtures to use in the mixture of logistics loss.
    """

    reconstruction_loss: Literal['dmol', 'mse'] = "dmol"
    width: int = 16
    custom_width_str: Optional[str] = "32:16,16:32,8:64,4:128,1:256"
    enc_blocks: str = "32x11,32d2,16x6,16d2,8x6,8d2,4x3,4d4,1x3"
    dec_blocks: str = "1x1,4m1,4x2,8m4,8x5,16m8,16x10,32m16,32x21"
    bottleneck_multiple: float = 0.25
    no_bias_above: int = 64
    num_mixtures: int = 10