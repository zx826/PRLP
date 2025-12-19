import math
from dataclasses import dataclass
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from mamba_ssm.modules.mamba2 import Mamba2
from FusionNet import StConv
@dataclass
class MambaConfig:
    d_model: int  # D
    n_layers: int
    input_size: int = 256
    # scale: int
    in_channels: int = 3
    out_channels: int = 3
    # hide_channels: int



class DualMamba(nn.Module):
    def __init__(self, config: MambaConfig, patch_size,layer):
        super().__init__()
        self.config = config
        self.input_size = config.input_size
        self.in_channels = config.in_channels
        self.x_embedder1 = PatchEmbed(config.input_size, patch_size, config.in_channels, config.d_model,bias=True)  # PatchEmbed函数就是patchify化,img_size是图像尺寸patch_size是每个patch的大小
        self.x_embedder2 = PatchEmbed(config.input_size, patch_size*2, config.in_channels, config.d_model,bias=True)  # PatchEmbed函数就是patchify化,img_size是图像尺寸patch_size是每个patch的大小

        self.layers1 = nn.ModuleList([MambaBlock(config) for _ in range(layer)])
        self.layers2 = nn.ModuleList([MambaBlock(config) for _ in range(layer)])

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.out_channels = config.out_channels
        self.pos_embed_x1 = nn.Parameter(torch.zeros(1, self.x_embedder1.num_patches, config.d_model), requires_grad=False)
        self.pos_embed_x2 = nn.Parameter(torch.zeros(1, self.x_embedder2.num_patches, config.d_model), requires_grad=False)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        self.final_layer1 = FinalLayer(config.d_model, patch_size, config.in_channels)
        self.final_layer2 = FinalLayer(config.d_model, patch_size*2, config.in_channels)
        self.FusionNet = nn.Conv2d(6, 3, kernel_size=1, bias=True)
        self.initialize_weights()

    '''初始化权重'''
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed_x1 = get_2d_sincos_pos_embed(self.pos_embed_x1.shape[-1], int(self.x_embedder1.num_patches ** 0.5))
        self.pos_embed_x1.data.copy_(torch.from_numpy(pos_embed_x1).float().unsqueeze(0))
        pos_embed_x2 = get_2d_sincos_pos_embed(self.pos_embed_x2.shape[-1], int(self.x_embedder2.num_patches ** 0.5))
        self.pos_embed_x2.data.copy_(torch.from_numpy(pos_embed_x2).float().unsqueeze(0))


        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder1.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder1.proj.bias, 0)

        w = self.x_embedder2.proj.weight.data
        nn.init.constant_(self.x_embedder2.proj.bias, 0)


        # Zero-out output layers:
        nn.init.constant_(self.final_layer1.linear.weight, 0)
        nn.init.constant_(self.final_layer1.linear.bias, 0)
        nn.init.constant_(self.final_layer2.linear.weight, 0)
        nn.init.constant_(self.final_layer2.linear.bias, 0)


    def unpatchify1(self, x):
        c = self.in_channels
        p = self.x_embedder1.patch_size[0] #2
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h*p, h*p))
        return imgs

    def unpatchify2(self, x):
        c = self.in_channels
        p = self.x_embedder2.patch_size[0] #2
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h*p, h*p))
        return imgs

    # def forward(self, x ,pred_xstart=None,model_zero=None,**kwargs):
    def forward(self, x):
        x1= self.x_embedder1(x) + self.pos_embed_x1  # (N, T, D), where T = H * W / patch_size ** 2    torch.Size([10, 256, 16])# x : (B, L, D)
        for layer in self.layers1:
            x1 = layer(x1)
        x1 = self.final_layer1(x1)  # (N, T, patch_size ** 2 * out_channels)
        x1= self.unpatchify1(x1)                   # (N, out_channels, H, W)

        x2= self.x_embedder2(x) + self.pos_embed_x2  # (N, T, D), where T = H * W / patch_size ** 2    torch.Size([10, 256, 16])# x : (B, L, D)
        for layer in self.layers2:
            x2 = layer(x2)
        x2 = self.final_layer2(x2)  # (N, T, patch_size ** 2 * out_channels)
        x2 = self.unpatchify2(x2)                   # (N, out_channels, H, W)

        x_1 = torch.cat([x1,x2],dim=1)
        out = self.FusionNet(x_1)
        return  out 


class Model(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.layers1 = nn.ModuleList([DualMamba(config, 2 ,6) for _ in range(config.n_layers)])

        self.layers2 = nn.ModuleList([DualMamba(config, 4 ,9) for _ in range(config.n_layers)])
        
        self.layers3 = nn.ModuleList([DualMamba(config, 2 ,6) for _ in range(config.n_layers)])

    def forward(self, x):
        for layer1 in self.layers1:
            x = layer1(x)
        for layer2 in self.layers2:
            x = layer2(x)
        for layer3 in self.layers3:
            x = layer3(x)
        out = x

        return  out



class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.Mamba = Mamba2(config.d_model)


    def forward(self, x):

        output = self.Mamba(x)


        return output

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)


    def forward(self, x):
        x = self.linear(x)

        return x                 # (N, T, patch_size ** 2 * out_channels)










def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed
def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

