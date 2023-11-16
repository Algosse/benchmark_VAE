from typing import Union, Tuple, Optional

from pydantic.dataclasses import dataclass
from typing_extensions import Literal

from pythae.config import BaseConfig

@dataclass
class SwinTransformerConfig(BaseConfig):
    """Hierarchical VAE config class.

    Parameters:
    """
    
    num_heads: Union[Tuple[int, ...], None]
    nb_blocks: Union[Tuple[int, ...], None]
    sequence_size: int = 10
    sequence_to_image: Literal["mean", "attention", "class_token"] = "mean"
    window_size: int = 4
    mlp_ratio: float = 4.
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    drop_rate: float = 0.
    attn_drop_rate: float = 0.
    drop_path_rate: float = 0.
    ape: bool = False
    embed_norm: bool = True
    use_checkpoint: bool = False
    fused_window_process=False
