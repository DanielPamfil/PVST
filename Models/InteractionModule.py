import torch
import torch.nn as nn
from timm.models.layers import DropPath


class InteractionBlock(nn.Module):
    """
    Interaction Block for integrating features from different levels or scales.

    This module is useful for tasks that require understanding and merging information from multiple feature
    representations, such as in computer vision models where both low-level and high-level features are important.

    Args:
        dim (int): Number of channels in the low-level feature set.
        dim1, dim2 (int): Number of channels in the high-level feature sets.
        embed_dim (int): Dimensionality for the attention mechanism.
        num_heads (int): Number of heads in the attention mechanism. This splits the attention into multiple heads to
            capture different types of relationships in parallel.
        mlp_ratio (float): Ratio of the hidden dimension size to the embedding dimension size in the
            MLP (multi-layer perceptron) layer.
    """

    def __init__(self, dim, dim1, dim2=None, embed_dim=384, num_heads=6, mlp_ratio=3., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(InteractionBlock, self).__init__()
        # Cross-attention modules for feature interaction
        self.interact1 = CrossAttention(dim1=dim, dim2=dim1, dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                        qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # Normalization layers for input features
        self.norm0 = norm_layer(dim)
        self.norm1 = norm_layer(dim1)
        self.dim = dim
        self.dim2 = dim2
        self.mlp_ratio = mlp_ratio
        # Additional cross-attention module if a third feature dimension is provided
        if self.dim2:
            self.interact2 = CrossAttention(dim1=dim, dim2=dim2, dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.norm2 = norm_layer(dim2)
        # # DropPath for regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm = nn.LayerNorm(dim)
        # MLP for further feature transformation
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            act_layer(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, fea, fea_1, fea_2=None):
        """
        Forward pass for the Interaction Block. It processes input features using cross-attention and an MLP.

        Args:
            fea: Tensor representing low-level features.
            fea_1: Tensor representing the first set of high-level features.
            fea_2: Optional tensor representing the second set of high-level features.

        Returns:
            Tensor: The output feature map tensor after integrating the features.
        """
        # Normalizing and applying cross-attention to the features
        fea = self.norm0(fea)
        fea_1 = self.norm1(fea_1)
        fea_1 = self.interact1(fea, fea_1)
        if self.dim2:
            fea_2 = self.norm2(fea_2)
            fea_2 = self.interact2(fea, fea_2)
        # Combining features and passing through MLP
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
    """
    Cross-attention Module.

    Args:
        dim1, dim2 (int): The dimensions of the first and second input feature sets.
        dim (int): The dimension of the query, key, and value in the attention mechanism.
        num_heads (int): The number of attention heads. Attention is split across these heads for parallel computation.
        qkv_bias (bool): If set to True, a learnable bias is added to the query, key, and value calculations.
        qk_scale (float, optional): An optional scaling factor for the query-key dot product. If not set,
            defaults to head_dim ** -0.5.
        attn_drop (float): Dropout rate applied to the attention weights to prevent overfitting.
        proj_drop (float): Dropout rate applied to the output of the attention module for regularization.
    """
    def __init__(self, dim1, dim2, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        # Number of attention heads and the dimension per head
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # Dimensions of the input feature sets
        self.dim = dim
        self.dim1 = dim1
        self.dim2 = dim2
        # Scale factor for the attention scores
        self.scale = qk_scale or head_dim ** -0.5

        # Linear layers for query, key, value, and projection
        self.q1 = nn.Linear(dim1, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim1)
        self.k2 = nn.Linear(dim2, dim, bias=qkv_bias)
        self.v2 = nn.Linear(dim2, dim, bias=qkv_bias)

        # Dropout layers for attention and output
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, fea, depth_fea):
        """
        Forward pass for the CrossAttention module.

        Args:
            fea: Tensor of shape (batch_size, N1, dim1), representing the first set of features.
            depth_fea: Tensor of shape (batch_size, N, dim2), representing the second set of features.

        Returns:
            Feature map: Tensor of shape (batch_size, N1, dim1), the result of applying cross-attention.
        """
        # Forward pass of the CrossAttention, computing attention between features
        _, N1, _ = fea.shape
        B, N, _ = depth_fea.shape
        C = self.dim
        q1 = self.q1(fea).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # q [B, nhead, N, C//nhead]

        k2 = self.k2(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v2 = self.v2(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Computing attention scores between the query from 'fea' and key from 'depth_fea'
        attn = (q1 @ k2.transpose(-2, -1)) * self.scale
        # Applying softmax to get the attention weights
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Applying attention to the value and projecting the result back to the original dimension
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