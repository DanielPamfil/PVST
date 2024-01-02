import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class decoder(nn.Module):
    """
    Decoder module for up-sampling and fusing multi-scale feature maps in a neural network.

    Attributes:
        embed_dim (int): The dimension size of embeddings used in the attention layers.
        dims (list[int]): A list specifying the number of channels in the feature maps at different scales.
        img_size (int): The size of the input image for which the feature maps are being decoded.
        mlp_ratio (float): Ratio of MLP (multi-layer perceptron) hidden dimension to embedding dimension,
        used in attention layers.

    The decoder consists of several multiscale fusion layers and global window attention layers to process and integrate
    features at different resolutions, and projects the fused features to output masks at different scales.
    """
    def __init__(self, embed_dim=512, dims=[64, 128, 320], img_size=224, mlp_ratio=3):
        super(decoder, self).__init__()
        # Store input parameters
        self.img_size = img_size
        self.dims = dims
        self.embed_dim = embed_dim

        # Fusion layers for combining features from different scales
        self.fusion1 = multiscale_fusion(in_dim=dims[2], f_dim=dims[1], kernel_size=(3, 3),
                                         img_size=(img_size // 8, img_size // 8), stride=(2, 2), padding=(1, 1))
        self.fusion2 = multiscale_fusion(in_dim=dims[1], f_dim=dims[0], kernel_size=(3, 3),
                                         img_size=(img_size // 4, img_size // 4), stride=(2, 2), padding=(1, 1))
        self.fusion3 = multiscale_fusion(in_dim=dims[0], f_dim=dims[0], kernel_size=(7, 7),
                                         img_size=(img_size // 1, img_size // 1), stride=(4, 4), padding=(2, 2),
                                         fuse=False)

        # Global window attention layers
        self.mixatt1 = GlobalWindowAttention(in_dim=dims[1], dim=embed_dim, img_size=(img_size // 8, img_size // 8),
                                             window_size=(img_size // 32), num_heads=1, mlp_ratio=mlp_ratio, depth=2)
        self.mixatt2 = GlobalWindowAttention(in_dim=dims[0], dim=embed_dim, img_size=(img_size // 4, img_size // 4),
                                             window_size=(img_size // 32), num_heads=1, mlp_ratio=mlp_ratio, depth=2)

        # Projection layers to map the features to mask outputs
        self.proj1 = nn.Linear(dims[2], 1)
        self.proj2 = nn.Linear(dims[1], 1)
        self.proj3 = nn.Linear(dims[0], 1)
        self.proj4 = nn.Linear(dims[0], 1)

    def forward(self, f):
        """
        Forward pass for the decoder. Processes input feature maps and produces output masks at various scales.

        Args:
           f (list of Tensors): Input feature maps at different scales.

        Returns:
           list of Tensors: Output masks at various scales corresponding to the input feature maps.
        """
        # Extracting feature maps at different scales
        fea_1_16, fea_1_8, fea_1_4 = f  # fea_1_16:1/16
        B, _, _ = fea_1_16.shape

        # Process feature maps through fusion and attention layers
        fea_1_8 = self.fusion1(fea_1_16, fea_1_8)
        fea_1_8 = self.mixatt1(fea_1_8)

        fea_1_4 = self.fusion2(fea_1_8, fea_1_4)
        fea_1_4 = self.mixatt2(fea_1_4)

        fea_1_1 = self.fusion3(fea_1_4)

        fea_1_16 = self.proj1(fea_1_16)

        # Project the processed features to masks
        mask_1_16 = fea_1_16.transpose(1, 2).reshape(B, 1, self.img_size // 16, self.img_size // 16)

        fea_1_8 = self.proj2(fea_1_8)
        mask_1_8 = fea_1_8.transpose(1, 2).reshape(B, 1, self.img_size // 8, self.img_size // 8)

        fea_1_4 = self.proj3(fea_1_4)
        mask_1_4 = fea_1_4.transpose(1, 2).reshape(B, 1, self.img_size // 4, self.img_size // 4)

        fea_1_1 = self.proj4(fea_1_1)
        mask_1_1 = fea_1_1.transpose(1, 2).reshape(B, 1, self.img_size // 1, self.img_size // 1)

        # Return masks at different scales
        return [mask_1_16, mask_1_8, mask_1_4, mask_1_1]

    def flops(self):
        flops = 0
        flops += self.fusion1.flops()
        flops += self.fusion2.flops()
        flops += self.fusion3.flops()
        flops += self.mixatt1.flops()
        flops += self.mixatt2.flops()

        flops += self.img_size // 16 * self.img_size // 16 * self.dims[2]
        flops += self.img_size // 8 * self.img_size // 8 * self.dims[1]
        flops += self.img_size // 4 * self.img_size // 4 * self.dims[0]
        flops += self.img_size // 1 * self.img_size // 1 * self.dims[0]

        return flops


class multiscale_fusion(nn.Module):
    """
    Module for up-sampling and fusing features from different scales.

    This module is useful when features extracted at different layers (or scales) need to be combined for tasks like
    segmentation (Salient Object Detection.

    Args:
        in_dim (int): Number of channels in the input feature map.
        f_dim (int): Number of channels in the fused feature map.
        kernel_size (tuple(int)): The size of the kernel used for up-sampling. It dictates how the up-sampling is
            performed.
        img_size (tuple(int)): The size (height, width) of the output image after up-sampling.
        stride (tuple(int)): The stride of the up-sampling operation. This determines how far apart the values are
            sampled in the input feature map.
        padding (tuple(int)): The amount of padding applied to the input feature map before up-sampling.
        fuse (bool): A flag to determine if features from different scales are concatenated (fused) or not.
    """
    def __init__(self, in_dim, f_dim, kernel_size, img_size, stride, padding, fuse=True):
        super(multiscale_fusion, self).__init__()
        # Initialization of module attributes
        # Determines whether to fuse additional features
        self.fuse = fuse
        # Normalization layer
        self.norm = nn.LayerNorm(in_dim)
        # Dimension of input features
        self.in_dim = in_dim
        # Dimension of output feature
        self.f_dim = f_dim
        # Kernel size for up-sampling
        self.kernel_size = kernel_size
        # Target image size after up-sampling
        self.img_size = img_size
        # Projection layer for feature transformation
        self.project = nn.Linear(in_dim, in_dim * kernel_size[0] * kernel_size[1])
        # Up-sampling layer
        self.upsample = nn.Fold(output_size=img_size, kernel_size=kernel_size, stride=stride, padding=padding)
        # Conditional layer initialization based on fusion flag
        if self.fuse:
            # Multi-layer perceptron for feature fusion
            self.mlp1 = nn.Sequential(
                nn.Linear(in_dim + f_dim, f_dim),
                nn.GELU(),
                nn.Linear(f_dim, f_dim),
            )
        else:
            # Projection layer when fusion is not required
            self.proj = nn.Linear(in_dim, f_dim)

    def forward(self, fea, fea_1=None):
        """
        Forward pass of the multiscale fusion module.

        Args:
           fea: Input feature map to be up-sampled.
           fea_1: Additional feature map to be fused with the up-sampled features. Used only if `fuse` is True.

        Returns:
           Tensor: The up-sampled (and potentially fused) feature map.
        """
        # Normalize and project the input features
        fea = self.project(self.norm(fea))
        # Perform up-sampling
        fea = self.upsample(fea.transpose(1, 2))
        B, C, _, _ = fea.shape
        # Reshape features for potential fusion
        fea = fea.view(B, C, -1).transpose(1, 2)  # .contiguous()
        # Fuse features if required
        if self.fuse:
            # Concatenate additional features
            fea = torch.cat([fea, fea_1], dim=2)
            # Apply MLP for feature fusion
            fea = self.mlp1(fea)
        else:
            # Project features if no fusion is required
            fea = self.proj(fea)
        return fea

    def flops(self):
        N = self.img_size[0] * self.img_size[1]
        flops = 0
        # norm
        flops += N * self.in_dim
        # proj
        flops += N * self.in_dim * self.in_dim * self.kernel_size[0] * self.kernel_size[1]
        # mlp
        flops += N * (self.in_dim + self.f_dim) * self.f_dim
        flops += N * self.f_dim * self.f_dim
        return flops


class GlobalWindowAttention(nn.Module):
    """
    Attention module integrating both global and window-based (local) attention.

    This module is designed to process feature maps in a way that captures both local features within certain windows
    and global dependencies across the entire image or feature map.

    Args:
        in_dim (int): The number of channels in the input feature map.
        dim (int): The dimensionality used for the internal attention calculations.
        img_size (int): The size of the image or feature map after up-sampling, used for determining the scope of
            global attention.
        window_size (int): The size of the window used for local attention. This defines the area in which local
            features are aggregated.
        num_heads (int): The number of attention heads. Multiple heads allow the model to simultaneously focus on
            different parts of the feature map.
        mlp_ratio (float): The ratio of the MLP (multi-layer perceptron) hidden layer size to the attention dimension
            size. This ratio impacts the capacity and complexity of the MLP layers used after attention operations.
        depth (int): The number of consecutive attention blocks in this module. More layers enable more complex
            feature extraction but also increase computational requirements.
    """

    def __init__(self, in_dim, dim, img_size, window_size, num_heads=1, mlp_ratio=4, depth=2, drop_path=0.):
        super(GlobalWindowAttention, self).__init__()
        # Store the dimensions and size parameters
        self.img_size = img_size
        self.in_dim = in_dim
        self.dim = dim
        # Normalization and initial MLP to transform the input features
        self.norm1 = nn.LayerNorm(in_dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        # Series of attention blocks that apply both global and window-based attention
        self.blocks = nn.ModuleList([
            GlobalWindowAttentionBlock(dim=dim, img_size=img_size, window_size=window_size, num_heads=num_heads,
                                       mlp_ratio=mlp_ratio)
            for i in range(depth)])
        # Normalization and final MLP to project the attention-modified features back to the original dimension
        self.norm2 = nn.LayerNorm(dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )
        # Drop path layer for regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, fea):
        """
        Forward pass of the GlobalWindowAttention module.

        Args:
            fea: The input feature map.

        Returns:
            Tensor: The output feature map after applying global and window-based attention.
        """
        # Process the input features through the initial MLP and normalization
        fea = self.mlp1(self.norm1(fea))
        # Pass the features through each attention block
        for blk in self.blocks:
            fea = blk(fea)
        # Apply the final MLP and normalization to the features, and return
        fea = self.drop_path(self.mlp2(self.norm2(fea)))
        return fea

    def flops(self):
        flops = 0
        N = self.img_size[0] * self.img_size[1]
        # norm1
        flops += N * self.in_dim
        # mlp1
        flops += N * self.in_dim * self.dim
        flops += N * self.dim * self.dim
        # blks
        for blk in self.blocks:
            flops += blk.flops()
        # norm2
        flops += N * self.dim
        # mlp2
        flops += N * self.in_dim * self.dim
        flops += N * self.dim * self.dim
        return flops

class GlobalWindowAttentionBlock(nn.Module):
    """
    Combines window-based and global attention mechanisms.

    This block is designed to process input features by applying both local (window-based) and global attention,
    providing a comprehensive understanding of the input data at different scales.

    Args:
        dim (int): The dimensionality of the input features and the internal representations in the attention mechanisms.
        img_size (int): The size of the input feature map (height and width).
        window_size (int): The size of the window used for local (window-based) attention.
        num_heads (int): The number of heads in the attention mechanism, allowing the model to focus on different parts
            of the input simultaneously.
        mlp_ratio (float): The ratio of the hidden layer size to the input size in the MLP following the attention layers.
        drop_path (float): The dropout probability used in the DropPath regularization.
    """
    def __init__(self, dim, img_size, window_size, num_heads=1, mlp_ratio=3, drop_path=0.):
        super(GlobalWindowAttentionBlock, self).__init__()
        # Store the input parameters
        self.img_size = img_size
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        # Window-based attention block
        self.windowatt = WindowAttentionBlock(dim=dim, input_resolution=img_size, num_heads=num_heads,
                                              window_size=window_size, shift_size=0,
                                              mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                                              drop_path=0.,
                                              act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                                              fused_window_process=False)
        # Global attention block
        self.globalatt = Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=False, qk_scale=None,
                               drop=0., attn_drop=0.,
                               drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        # Normalization and MLP for further processing the features
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        # DropPath layer for regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """
        Forward pass of the GlobalWindowAttentionBlock.

        Args:
            x: The input feature map to the block.

        Returns:
            Tensor: The output feature map after applying both window-based and global attention.
        """
        # Apply window-based attention and global attention
        att1 = self.windowatt(x)
        att2 = self.globalatt(x)
        # Combine the results from both attention mechanisms and pass through MLP
        x = x + att1 + att2
        x = x + self.drop_path(self.mlp(self.norm(x)))
        return x

    def flops(self):
        N = self.img_size[0] * self.img_size[1]
        flops = 0
        flops += self.windowatt.flops()
        flops += self.globalatt.flops(N)
        flops += self.dim * N
        flops += 2 * N * self.dim * self.dim * self.mlp_ratio
        return flops

class WindowAttentionBlock(nn.Module):
    """
    Window-based attention block adapted from Swin Transformer Block.

    This block specializes in applying attention within localized windows of an input feature map.
    It's particularly effective in contexts where capturing local details is essential, such as in image processing tasks.

    Args:
        dim (int): The number of input channels or the dimensionality of feature vectors at each position.
        input_resolution (tuple(int)): The height and width of the input feature map.
        num_heads (int): The number of attention heads in the multi-head attention mechanism.
        window_size (int): The size of the square window for applying attention.
        shift_size (int): The shift size for cyclic window partitioning, enabling cross-window connections.
        mlp_ratio (float): The ratio of the MLP hidden layer dimension to the dimension of the input layer.
        qkv_bias (bool): If True, adds a learnable bias to query, key, value in the attention mechanism.
        qk_scale (float, optional): Scaling factor for query-key dot products in the attention mechanism.
        drop, attn_drop (float): Dropout rates for the attention outputs and the final output of the block.
        drop_path (float): Dropout rate for DropPath regularization.
        act_layer, norm_layer (nn.Module): Types of activation and normalization layers to use.
        fused_window_process (bool): Flag indicating whether to use a fused process for window-based attention.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        # Store parameters and configurations
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # Normalization layer
        self.norm1 = norm_layer(dim)
        # Window Attention Mechanism
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # DropPath for regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Calculate attention mask for SW-MSA (Shifted Window Multi-Head Self-Attention)
        if self.shift_size > 0:
            # Mask calculation for SW-MSA to enable cross-window connection
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
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
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        # Register attention mask as a buffer
        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        """
        Forward pass of the WindowAttentionBlock.

        Applies window-based attention to the input feature map, taking into account both local window features and
        shifted window features for broader context.

        Args:
            x (Tensor): The input feature map.

        Returns:
            Tensor: Output feature map after applying window-based attention.
        """
        # Validate input dimensions
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        # Apply normalization
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Apply cyclic shift and partition into windows
        if self.shift_size > 0:
            # Shift and partition for SW-MSA
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        else:
            # Partition for standard window-based attention
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # Perform Window-based Multi-Head Self-Attention (W-MSA or SW-MSA)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # Merge window outputs
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # Apply reverse cyclic shift if needed
        if self.shift_size > 0:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = self.drop_path(x)


        return x



class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention module with relative position bias.

    This module is designed to apply self-attention within a local window of inputs, allowing the model to focus on and
    integrate local contextual information efficiently. It is particularly useful in tasks where local patterns within a
    specific area are more relevant than distant interactions.

    Args:
        dim (int): The number of input channels.
        window_size (tuple[int]): The height and width of the attention window, defining the local area within which
            attention is computed.
        num_heads (int): The number of attention heads, allowing the model to focus on different aspects of the input
            simultaneously.
        qkv_bias (bool, optional): If True, adds a learnable bias to the query, key, and value tensors in the attention
            mechanism.
        qk_scale (float | None, optional): A scaling factor for the dot-product in the attention calculation. If None,
            defaults to the inverse square root of the head dimension.
        attn_drop (float, optional): Dropout ratio for the attention weights, providing regularization.
        proj_drop (float, optional): Dropout ratio for the output of the attention module, providing another layer of
            regularization.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        # Initialize dimensions and configurations
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Initialize the relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # Compute relative position index
        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        # Layers for computing query, key, and value in the attention mechanism
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # Layers for computing query, key, and value in the attention mechanism
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Forward pass of the WindowAttention module.

        Args:
            x: The input features, expected to be of shape (num_windows * B, N, C), where B is the batch size, N is
                the number of tokens in each window, and C is the number of channels.
            mask: An optional attention mask, of shape (num_windows, window_height * window_width,
                window_height * window_width), used to mask out certain positions within each window.

        Returns:
            The output features after applying window-based attention, of the same shape as the input.
        """
        # Determine shape of input features
        B_, N, C = x.shape
        # Linearly project the input features to get queries, keys, and values
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # Scale the query vectors
        q = q * self.scale
        # Compute the attention scores using dot-product of queries and keys
        attn = (q @ k.transpose(-2, -1))
        # Add the relative position bias to the attention scores
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        # Apply the attention mask if provided
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        # Apply dropout to the attention weights
        attn = self.attn_drop(attn)
        # Compute the output features as a weighted sum of the values
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # Project the output features and apply dropout
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

def window_partition(x, window_size):
    """
    Partitions an input tensor into smaller windowed tensors.

    This function is used to divide a larger tensor into smaller square windows. Each window is of size
    'window_size x window_size'. This is typically used in attention mechanisms where local context within these windows
     is important.

    Args:
        x (Tensor): A 4D input tensor with shape (B, H, W, C), where B is the batch size, H and W are the height and
            width of the feature map, and C is the number of channels.
        window_size (int): The size of each square window.

    Returns:
        Tensor: A reshaped tensor where the input is divided into windows. The shape of the output tensor is
        (num_windows * B, window_size, window_size, C). Here, num_windows is the number of windows formed from the input
        tensor.
    """
    B, H, W, C = x.shape
    # Reshape and permute the tensor to form windows
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # Rearrange the tensor dimensions to get the windowed tensor
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Reverses the window partitioning, reconstructing the original tensor.

    This function takes windowed tensors and reassembles them into the original larger tensor.

    Args:
        windows (Tensor): A tensor containing windowed features, with shape
            (num_windows * B, window_size, window_size, C).
        window_size (int): The size of each square window.
        H (int): The height of the original (larger) tensor before window partitioning.
        W (int): The width of the original (larger) tensor before window partitioning.

    Returns:
        Tensor: A reconstructed tensor with shape (B, H, W, C), where B is the batch size and C is the number of channels.
        This tensor has the same shape as the original tensor before window partitioning.
    """
    # Calculate the batch size based on the number of windows and the dimensions of the original image
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # Reshape and permute the windows to reconstruct the original tensor layout
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # Rearrange the tensor dimensions to get the original tensor shape
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Block(nn.Module):
    """
    Building block implementing Global self-attention mechanism.

    This class represents a simplified version of a transformer block, focusing on the self-attention part
    and excluding the typical feedforward network found in standard transformer blocks.
    It's suitable for tasks where capturing global dependencies within the data is crucial.

    Args:
        dim (int): The number of input channels or the dimensionality of the feature vectors at each position in the
            input.
        num_heads (int): The number of attention heads in the multi-head attention mechanism.
        mlp_ratio (float): The ratio of the hidden layer size to the input size in the MLP. This argument is not
            used in the current implementation.
        qkv_bias (bool): If True, adds a learnable bias to the query, key, and value tensors in the attention
            mechanism.
        qk_scale (float | None, optional): A scaling factor for the dot-product in the attention calculation.
        drop (float): Dropout rate for the output of the attention layer.
        attn_drop (float): Dropout rate for the attention weights.
        drop_path (float): Dropout rate for DropPath regularization, providing a form of stochastic depth.
        act_layer (nn.Module): The type of activation layer to use. Not used in the current implementation.
        norm_layer (nn.Module): The type of normalization layer to use. Typically LayerNorm.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # Normalization layer
        self.norm1 = norm_layer(dim)
        # Store the dimension
        self.dim = dim
        # Attention mechanism
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # DropPath for regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x):
        """
        Forward pass of the Block.

        Args:
            x (Tensor): The input tensor to the block.

        Returns:
            Tensor: The output tensor after applying normalization and self-attention.
        """
        # Apply normalization, self-attention and DropPath
        x = self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def flops(self, N):
        flops = 0
        # att
        flops += self.attn.flops(N)
        # norm
        flops += self.dim * N
        return flops

class Attention(nn.Module):
    """
    Multi-head self-attention mechanism.

    This class implements the multi-head self-attention, a key component in transformer models. It allows the model to
    weigh the importance of different parts of the input sequence differently, enabling it to capture complex
    dependencies.

    Args:
        dim (int): The dimensionality of the input features. Also the output dimension.
        num_heads (int): The number of attention heads. Multi-head attention allows the model to jointly attend to
            information from different representation subspaces at different positions.
        qkv_bias (bool): If set to True, adds a learnable bias to query, key, and value projections.
        qk_scale (float, optional): A scaling factor for the dot-product in the attention calculation. If None, uses
            1/sqrt(head_dim).
        attn_drop (float): Dropout ratio for attention weights.
        proj_drop (float): Dropout ratio for the output of the attention block.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        # Scaling factor for attention scores
        self.scale = qk_scale or head_dim ** -0.5

        # Linear layers for projecting the input to queries, keys, and values
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # Dropout layers for attention weights and the final output
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (Tensor): Input tensor of shape (B, N, C) where B is the batch size, N is the sequence length, and C is the number of channels (feature dimension).

        Returns:
            Tensor: Output tensor after applying self-attention.
        """
        B, N, C = x.shape
        # Splitting the input into queries, keys, and values
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Calculating attention scores and applying softmax
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # Applying dropout to attention scores
        attn = self.attn_drop(attn)

        # Weighted sum of values and projecting back to the original dimension
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def flops(self, N):
        flops = 0
        # q
        flops += N * self.dim * self.dim * 3
        # qk
        flops += self.num_heads * N * self.dim // self.num_heads * N
        # att v
        flops += self.num_heads * N * self.dim // self.num_heads * N
        # proj
        flops += N * self.dim * self.dim
        return flops