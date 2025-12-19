import math
from dataclasses import dataclass
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from mamba_ssm.modules.mamba2 import Mamba2

@dataclass
class MambaConfig:
    d_model: int  # D
    n_layers: int
    input_size: int = 256
    # scale: int
    in_channels: int = 3
    out_channels: int = 3
    # hide_channels: int


class Downsample(nn.Module):
    def __init__(self, config: MambaConfig, in_channels, out_channels,input_size,patch_size,layers):
        super().__init__()
        #patch_size = 2
        self.in_channels = config.in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.x_embedder = PatchEmbed(input_size, patch_size, out_channels, config.d_model,bias=True)  # PatchEmbed函数就是patchify化,img_size是图像尺寸patch_size是每个patch的大小
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches, config.d_model), requires_grad=False)
        self.layers = nn.ModuleList([MambaBlock(config) for _ in range(layers)])

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        self.final_layer = FinalLayer(config.d_model, patch_size, out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed_x = get_2d_sincos_pos_embed(self.pos_embed_x.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed_x.data.copy_(torch.from_numpy(pos_embed_x).float().unsqueeze(0))


        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Zero-out output layers:

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        
    def unpatchify(self, x):
        c = self.in_channels
        p = self.x_embedder.patch_size[0] #2
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h*p, h*p))
        return imgs
        
    def forward(self, x):
        x = self.conv(x)  # [B, C, H, W] -> [B, C', H/2, W/2]
        # x = x.permute(0, 2, 3, 1)  # [B, C', H, W] -> [B, H, W, C']
        x = self.norm(x)
        # x = x.permute(0, 3, 1, 2)  # [B, C', H, W]
        x = self.act(x)
        x = self.x_embedder(x) + self.pos_embed_x  # (N, T, D), where T = H * W / patch_size ** 2    torch.Size([10, 256, 16])# x : (B, L, D)
        for layer in self.layers:
            output = layer(x)
        x = self.final_layer(output)  # (N, T, patch_size ** 2 * out_channels)
        out = self.unpatchify(x)                   # (N, out_channels, H, W)
        
        return out



class Upsample(nn.Module):
    def __init__(self, config: MambaConfig, in_channels, out_channels,input_size,patch_size,layers):
        super().__init__()
        #patch_size = 2
        self.in_channels = config.in_channels
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.x_embedder = PatchEmbed(input_size, patch_size, out_channels*2, config.d_model,bias=True)  # PatchEmbed函数就是patchify化,img_size是图像尺寸patch_size是每个patch的大小
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches, config.d_model), requires_grad=False)
        self.layers = nn.ModuleList([MambaBlock(config) for _ in range(layers)])

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        self.final_layer = FinalLayer(config.d_model, patch_size, out_channels)
        self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed_x = get_2d_sincos_pos_embed(self.pos_embed_x.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed_x.data.copy_(torch.from_numpy(pos_embed_x).float().unsqueeze(0))


        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Zero-out output layers:

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        
    def unpatchify(self, x):
        c = self.in_channels
        p = self.x_embedder.patch_size[0] #2
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h*p, h*p))
        return imgs
        
    def forward(self, x1, x2):
        x1 = self.conv(x1)  # [B, C, H, W] -> [B, C', 2H, 2W] 
        x1 = self.norm(x1)
        x1 = self.act(x1)
        x = torch.cat([x1,x2],dim=1)
        x = self.x_embedder(x) + self.pos_embed_x  # (N, T, D), where T = H * W / patch_size ** 2    torch.Size([10, 256, 16])# x : (B, L, D)
        for layer in self.layers:
            output = layer(x)
        x = self.final_layer(output)  # (N, T, patch_size ** 2 * out_channels)
        out = self.unpatchify(x)                   # (N, out_channels, H, W)
        
        return out


class UMamba(nn.Module):
    def __init__(self, config: MambaConfig, patch_size=1):
        super().__init__()
        self.config = config
        self.input_size = config.input_size
        self.in_channels = config.in_channels
        self.x_embedder1 = PatchEmbed(32, patch_size, config.in_channels, config.d_model,bias=True)  # PatchEmbed函数就是patchify化,img_size是图像尺寸patch_size是每个patch的大小
        # self.x_embedder2 = PatchEmbed(32, patch_size, config.in_channels, config.d_model,bias=True)  # PatchEmbed函数就是patchify化,img_size是图像尺寸patch_size是每个patch的大小
        self.x_embedder3 = PatchEmbed(32, patch_size, config.in_channels, config.d_model,bias=True)  # PatchEmbed函数就是patchify化,img_size是图像尺寸patch_size是每个patch的大小
        
        self.layers1 = nn.ModuleList([MambaBlock(config) for _ in range(config.n_layers)])
        # self.layers2 = nn.ModuleList([MambaBlock(config) for _ in range(config.n_layers)])
        self.layers3 = nn.ModuleList([MambaBlock(config) for _ in range(config.n_layers)])
        
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.out_channels = config.out_channels
        self.pos_embed_x1 = nn.Parameter(torch.zeros(1, self.x_embedder1.num_patches, config.d_model), requires_grad=False)
        # self.pos_embed_x2 = nn.Parameter(torch.zeros(1, self.x_embedder2.num_patches, config.d_model), requires_grad=False)
        self.pos_embed_x3 = nn.Parameter(torch.zeros(1, self.x_embedder3.num_patches, config.d_model), requires_grad=False)
        self.down1 = Downsample(config,config.in_channels,self.out_channels,128,8,8)
        self.down2 = Downsample(config,config.in_channels,self.out_channels,64,2,8)
        self.down3 = Downsample(config,config.in_channels,self.out_channels,32,1,8)
        self.up1 = Upsample(config,config.in_channels,self.out_channels,64,1,8)
        self.up2 = Upsample(config,config.in_channels,self.out_channels,128,2,8)
        self.up3 = Upsample(config,config.in_channels,self.out_channels,256,2,8)
        
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        self.final_layer1 = FinalLayer(config.d_model, patch_size, config.in_channels)
        # self.final_layer2 = FinalLayer(config.d_model, patch_size, config.in_channels)
        self.final_layer3 = FinalLayer(config.d_model, patch_size, config.in_channels)
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
        # pos_embed_x2 = get_2d_sincos_pos_embed(self.pos_embed_x2.shape[-1], int(self.x_embedder2.num_patches ** 0.5))
        # self.pos_embed_x2.data.copy_(torch.from_numpy(pos_embed_x2).float().unsqueeze(0))
        pos_embed_x3 = get_2d_sincos_pos_embed(self.pos_embed_x3.shape[-1], int(self.x_embedder3.num_patches ** 0.5))
        self.pos_embed_x3.data.copy_(torch.from_numpy(pos_embed_x3).float().unsqueeze(0))


        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder1.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder1.proj.bias, 0)
        
        # w = self.x_embedder2.proj.weight.data
        # nn.init.constant_(self.x_embedder2.proj.bias, 0)
        
        w = self.x_embedder3.proj.weight.data
        nn.init.constant_(self.x_embedder3.proj.bias, 0)

        # Zero-out output layers:

        nn.init.constant_(self.final_layer1.linear.weight, 0)
        nn.init.constant_(self.final_layer1.linear.bias, 0)
        # nn.init.constant_(self.final_layer2.linear.weight, 0)
        # nn.init.constant_(self.final_layer2.linear.bias, 0)
        nn.init.constant_(self.final_layer3.linear.weight, 0)
        nn.init.constant_(self.final_layer3.linear.bias, 0)

        # nn.init.normal_(self.label_embedder.embedding_table.weight, std=0.02)
    def unpatchify1(self, x):
        c = self.in_channels
        p = self.x_embedder1.patch_size[0] #2
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h*p, h*p))
        return imgs
  
    def unpatchify3(self, x):
        c = self.in_channels
        p = self.x_embedder3.patch_size[0] #2
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h*p, h*p))
        return imgs
    def forward(self, x ,pred_xstart=None,model_zero=None,**kwargs):      
        x1 = self.down1(x)  
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        
        x3 = self.x_embedder1(x3) + self.pos_embed_x1  # (N, T, D), where T = H * W / patch_size ** 2    torch.Size([10, 256, 16])# x : (B, L, D)
        for layer in self.layers1:
            x32 = layer(x3)
        x32 = self.final_layer1(x32)  # (N, T, patch_size ** 2 * out_channels)
        x32 = self.unpatchify1(x32)                   # (N, out_channels, H, W)

        x33 = self.x_embedder3(x32) + self.pos_embed_x3  # (N, T, D), where T = H * W / patch_size ** 2    torch.Size([10, 256, 16])# x : (B, L, D)
        for layer in self.layers3:
            x33 = layer(x33)
        x33 = self.final_layer3(x33)  # (N, T, patch_size ** 2 * out_channels)
        x33 = self.unpatchify3(x33)                   # (N, out_channels, H, W)

        x_3 = self.up1(x33,x2) 
        x_2 = self.up2(x_3,x1)
        out = self.up3(x_2,x)
        
        
        return out

    # def step(self, x, caches):


    #     for i, layer in enumerate(self.layers):
    #         x, caches[i] = layer.step(x, caches[i])

    #     return x, caches



class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.Mamba = Mamba2(config.d_model)
        
    
    def forward(self, x):

        output = self.Mamba(x) + x 


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

