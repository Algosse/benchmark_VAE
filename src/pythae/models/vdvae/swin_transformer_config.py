import torch.nn as nn

from typing import Union, Tuple

from pydantic.dataclasses import dataclass
from typing_extensions import Literal

from pythae.config import BaseConfig

@dataclass
class SwinTransformerConfig(BaseConfig):
    """

    Parameters:
    """
    
    pretrained = None,
    patch_size: Tuple[int, int, int] = (1,1,1),
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=(2,7,7),
    mlp_ratio=4.,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.,
    attn_drop_rate=0.,
    drop_path_rate=0.2,
    norm_layer=nn.LayerNorm,
    patch_norm=False,
    frozen_stages=-1,
    use_checkpoint=False,
    sequence_to_image='mean',
    output_type='image',
    n_embed=3