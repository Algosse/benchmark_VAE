from typing import Union, Tuple

from pydantic.dataclasses import dataclass
from typing_extensions import Literal

from pythae.config import BaseConfig

@dataclass
class SCVAEConfig(BaseConfig):
    """Simple Conditionned VAE config class.

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
    """
    
    features: Union[Tuple[int, ...], None]
    nb_blocks: Union[Tuple[int, ...], None]
    dataset_name: str = 'SIARmini'
    input_dim: Union[Tuple[int, ...], None] = None
    latent_dim: int = 10
    reconstruction_loss: Literal["bce", "mse", "gaussian"] = "mse"
    uses_default_encoder: bool = False
    uses_default_decoder: bool = False