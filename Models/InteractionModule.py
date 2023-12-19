import torch
import torch.nn as nn
from timm.models.layers import DropPath


class InteractionBlock(nn.Module):
    r""" Interaction Block.

    Args:
        dim (int): Number of low-level feature channels.
        dim1, dim2 (int): Number of high-level feature channels.
        embed_dim (int): Dimension for attention.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
    """

    def __init__(self, dim, dim1, dim2=None, embed_dim=384, num_heads=6, mlp_ratio=3., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(InteractionBlock, self).__init__()
        self.interact1 = CrossAttention(dim1=dim, dim2=dim1, dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                        qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm0 = norm_layer(dim)
        self.norm1 = norm_layer(dim1)
        self.dim = dim
        self.dim2 = dim2
        self.mlp_ratio = mlp_ratio
        if self.dim2:
            self.interact2 = CrossAttention(dim1=dim, dim2=dim2, dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.norm2 = norm_layer(dim2)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            act_layer(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, fea, fea_1, fea_2=None):
        fea = self.norm0(fea)
        fea_1 = self.norm1(fea_1)
        fea_1 = self.interact1(fea, fea_1)
        if self.dim2:
            fea_2 = self.norm2(fea_2)
            fea_2 = self.interact2(fea, fea_2)
        fea = fea + fea_1
        if self.dim2:
            fea = fea + fea_2
        fea = fea + self.drop_path(self.mlp(self.norm(fea)))
        return fea

    def flops(self, N1, N2, N3=None):
        flops = 0
        flops += self.interact1.flops(N1, N2)
        if N3:
            flops += self.interact2.flops(N1, N3)
        flops += self.dim * N1
        flops += 2 * N1 * self.dim * self.dim * self.mlp_ratio
        return flops


class CrossAttention(nn.Module):
    def __init__(self, dim1, dim2, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        self.dim1 = dim1
        self.dim2 = dim2
        self.scale = qk_scale or head_dim ** -0.5

        self.q1 = nn.Linear(dim1, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim1)

        self.k2 = nn.Linear(dim2, dim, bias=qkv_bias)
        self.v2 = nn.Linear(dim2, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, fea, depth_fea):
        _, N1, _ = fea.shape
        B, N, _ = depth_fea.shape
        C = self.dim
        q1 = self.q1(fea).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # q [B, nhead, N, C//nhead]

        k2 = self.k2(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v2 = self.v2(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        fea = (attn @ v2).transpose(1, 2).reshape(B, N1, C)
        fea = self.proj(fea)
        fea = self.proj_drop(fea)

        return fea

    def flops(self, N1, N2):
        flops = 0
        # q
        flops += N1 * self.dim1 * self.dim
        # kv
        flops += N2 * self.dim2 * self.dim * 2
        # qk
        flops += self.num_heads * N1 * self.dim // self.num_heads * N2
        # att v
        flops += self.num_heads * N1 * self.dim // self.num_heads * N2
        # proj
        flops += N1 * self.dim * self.dim1
        return flops