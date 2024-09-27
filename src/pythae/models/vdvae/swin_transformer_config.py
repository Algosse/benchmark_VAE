from typing import Union, Tuple, Optional

from pydantic.dataclasses import dataclass
from typing_extensions import Literal

from pythae.config import BaseConfig

@dataclass
class SwinTransformerConfig(BaseConfig):
    """

    Parameters:
    """
    
    depths: Union[Tuple[int, ...], None]
    num_heads: Union[Tuple[int, ...], None]
    pretrained: bool = None
    patch_size: Tuple[int, int, int] = (2, 1, 1)
    embed_dim: int = 96
    window_size = (2,7,7)
    mlp_ratio: float = 4.
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    drop_rate: float = 0.
    attn_drop_rate: float = 0.
    drop_path_rate: float = 0.2
    norm_layer: Literal['layernorm'] = "layernorm"
    patch_norm: bool = False
    frozen_stages: int = -1
    use_checkpoint: bool = False
    sequence_to_image: str = 'mean'
    output_type: str = 'image'
    n_embed: int = 3