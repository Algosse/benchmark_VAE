# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Original: https://github.com/microsoft/Swin-Transformer/blob/e43ac64ce8abfe133ae582741ccaf6761eea05f7/models/swin_transformer.py
#
# Modified by: Alban Gosset
# For: Master Thesis at TU Berlin, Germany (2023)
#
# Deals with sequence of images instead of single images
# The images in a sequence does not have specific relations (neither temporal nor spatial)
# They are from http://arxiv.org/abs/2305.12036 dataset
#
# --------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
import torch.utils.checkpoint as checkpoint

from ..hievae_config import HieVAEConfig
from .swin_transformer_config import SwinTransformerConfig

from pythae.models.nn.base_architectures import BaseEncoder
from pythae.models.base.base_utils import ModelOutput


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size, S_window_size=None):
    """
    Args:
        x: (B, S, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, S, window_size, window_size, C)
    """    
    B, S, H, W, C = x.shape
    
    if S_window_size is None:
        # If the window size is not given, put the entire sequence in one window
        S_window_size = S
    
    x = x.view(B, S//S_window_size, S_window_size, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, S_window_size, window_size, window_size, C)
    return windows




def window_reverse(windows, window_size, H, W, S_window_size=None):
    """
    Args:
        windows: (num_windows*B, S, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, S, H, W, C)
    """
    S = windows.shape[1]
    if S_window_size is None:
        S_window_size = S
    
    B = int(windows.shape[0] / (H * W * S / window_size / window_size / S_window_size))
    x = windows.view(B, S // S_window_size, H // window_size, W // window_size, S, window_size, window_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, S, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    
    The positional encoding is done only on spatial dimensions. Not on the sequence dimension because the images does not have any temporal nor spatial relations.
    That is why the positional encoding is the same as the original code. It is only repeated along the sequence dimension.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, sequence_size, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * sequence_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1 * 2*S-1, nH

        # get pair-wise relative position index for each token inside the window
        # taken from https://github.com/haofanwang/video-swin-transformer-pytorch/blob/main/video_swin_transformer.py
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_s = torch.arange(sequence_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_s]))  # 3, Wh, Ww, S
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww*S, Wh*Ww*S
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1  # shift to start from 0
        relative_coords[:, :, 2] += sequence_size - 1        # shift to start from 0
        
        relative_coords[:, :, 1] *= 2 * self.window_size[1] - 1
        relative_coords[:, :, 2] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[0] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww*S, Wh*Ww*S
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        S = N // (self.window_size[0] * self.window_size[1])
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * S, self.window_size[0] * self.window_size[1] * S, -1)  # Wh*Ww*S,Wh*Ww*S,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww*S, Wh*Ww*S
        
        # Repeat the same positional encoding along the sequence dimension
        #relative_position_bias = relative_position_bias.repeat(1, S, S)
        
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlockSequence(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, sequence_size, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.sequence_size = sequence_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, sequence_size=sequence_size,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, 1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0)) # nW, Wh*Ww, Wh*Ww
            
            # repeat the same mask along the sequence dimension because we shift the windows only along spatial dimensions
            attn_mask = attn_mask.repeat(1, self.sequence_size, self.sequence_size)
            
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        H, W = self.input_resolution
        S = self.sequence_size
        B, L, C = x.shape
        assert L == H * W * S, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, S, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            # only unshift the spatial dimensions because all the images in the sequence are in one window
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x = x
            
        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size * S, C)  # nW*B, window_size*window_size*sequence_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size*sequence_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, S, self.window_size, self.window_size, C)

        # reverse windows
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B S H W C

        # reverse cyclic shift
        if self.shift_size > 0:
            # only unshift the spatial dimensions because all the images in the sequence are in one window
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        else:
            x = shifted_x
            
        x = x.view(B, S * H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, sequence_size={self.sequence_size}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    
    Only merge in the spatial dimension. The number of image in the sequence is not changed because we keep all the images in the same window.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, sequence_size, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.sequence_size = sequence_size
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W*S, C
        """
        H, W = self.input_resolution
        S = self.sequence_size
        B, L, C = x.shape
        assert L == H * W * S, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, S, H, W, C)

        x0 = x[:, :, 0::2, 0::2, :]  # B S H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B S H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B S H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B S H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B S H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B S*H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}, sequence_size={self.sequence_size}"


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, sequence_size, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.sequence_size = sequence_size
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlockSequence(dim=dim, input_resolution=input_resolution, sequence_size=sequence_size,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process)
            for i in range(depth)])

        # patch merging layer
        # Because merging is moved to the beggingin of the block, the input resolution and dimension have to be multiplied and divided by a factor of 2
        # Because it corresponds to the values of the previous block
        if downsample is not None:
            self.downsample = downsample((input_resolution[0]*2, input_resolution[1]*2), sequence_size=sequence_size, dim=dim//2, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        # Unlike the official implementation, Patch merging is applied at the beginning of the block.
        # It is not applied at the first block.
        if self.downsample is not None:
            x = self.downsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, sequence_size={self.sequence_size}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class LinearEmbed(nn.Module):
    r""" Linear embedding the sequence of images.
        Does not perform patching because we want to keep their full resolution.

    Args:
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Linear(in_chans, embed_dim)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, S, C, H, W = x.shape
        
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # B S H W C
        x = x.view(B, S * H * W, C)
        x = self.proj(x)

        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SequenceSwinTransformer(BaseEncoder):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
        
        The official implementation is modified to deal with sequence of images instead of single images.

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, model_config: HieVAEConfig, transformer_config: SwinTransformerConfig, **kwargs):
        super().__init__()
        
        img_size = model_config.input_dim[1]
        self.img_size = img_size
        sequence_size = transformer_config.sequence_size
        self.sequence_size = sequence_size
        nb_blocks = transformer_config.nb_blocks
        self.num_layers = len(nb_blocks)
        features = model_config.features
        self.num_features = len(features)
        self.ape = transformer_config.ape
        self.embed_norm = transformer_config.embed_norm
        self.mlp_ratio = transformer_config.mlp_ratio
        self.sequence_to_image = transformer_config.sequence_to_image
        
        in_chans = model_config.input_dim[0]
        norm_layer = torch.nn.LayerNorm
        drop_rate = transformer_config.drop_rate
        drop_path_rate = transformer_config.drop_path_rate
        num_heads = transformer_config.num_heads
        window_size = transformer_config.window_size
        qkv_bias = transformer_config.qkv_bias
        qk_scale = transformer_config.qk_scale
        attn_drop_rate = transformer_config.attn_drop_rate
        use_checkpoint = transformer_config.use_checkpoint
        fused_window_process = transformer_config.fused_window_process
        
        # split image into non-overlapping patches
        self.patch_embed = LinearEmbed(
            in_chans=in_chans, embed_dim=features[0],
            norm_layer=norm_layer if self.embed_norm else None)
        num_patches = img_size * img_size
        self.patches_resolution = img_size

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, features[0]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(nb_blocks))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=features[i_layer],
                               input_resolution=(img_size // (2 ** i_layer),
                                                 img_size // (2 ** i_layer)),
                               sequence_size=sequence_size,
                               depth=nb_blocks[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(nb_blocks[:i_layer]):sum(nb_blocks[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer > 0) else None,
                               use_checkpoint=use_checkpoint,
                               fused_window_process=fused_window_process)
            self.layers.append(layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, y, x=None):
        # Forward path for the SequenceSwinTransformer as a HVAE encoder
        # args:
        #   y (torch.tensor): B S C_in H W, the image sequences to deal with
        #   x (torch.tensor): B C_in H W, the ground truth image used for the posterior
        # returns:
        #   activations (dict): A dictionary containing the activations of the different layers of the encoder. 
        #                       The keys are the resolution of the image at the end of each blocks.
        
        activations = {}   
        
        if x is None:
            x = y
        else:
            x = torch.concat((x.unsqueeze(1), y), dim=1) # B 1+S C_in H W        
              
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for i, layer in enumerate(self.layers):
            x = layer(x) # B S*H/n*W/n C
            
            res = self.img_size // (2 ** i)
            act = x.view(x.shape[0], self.sequence_size, res, res, x.shape[2])
            act = act.permute(0, 1, 4, 2, 3).contiguous() # B S C H W
            
            act = self.get_image_from_sequence(act)
            
            activations[act.shape[2]] = act

        out = ModelOutput()
        out['activations'] = activations
        return out
    
    def get_image_from_sequence(self, act):
        """ Compute the activation based on the configuration of the model.
                Decoder excepts the activations to be in the form B C H W
                Need a way to deal with the sequence dimension
                Possibilities: mean along sequence dim, use a class token image (like in ViT), ...
        """
        
        if self.sequence_to_image == 'mean':
            out = act.mean(dim=1)
        elif self.sequence_to_image == 'attention':
            # compute an attention mask and apply it to the sequence to create a weighted sum of the inputs
            att_mask = torch.softmax(act, dim=1)
            out = (act * att_mask).sum(dim=1)
        elif self.sequence_to_image == 'class_token':
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown sequence_to_image value: {self.sequence_to_image}")
        
        return out

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

if __name__ == '__main__':
    
    sequence_size = 10
    img_size = 64
    channel_in = 3
    
    x = torch.randn(2, sequence_size, channel_in, img_size, img_size) # B S C_in H W
    
    features = [8, 16, 32, 64]
    nb_blocks = [2, 2, 2, 2]   
    
    model = SequenceSwinTransformer(img_size=img_size, sequence_size=sequence_size, in_chans=channel_in,
                                    features = features, nb_blocks = nb_blocks)
    
    activations = model(x)
    
    for key, value in activations.items():
        print(key, value.shape)
    
    