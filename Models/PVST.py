import torch
import torch.nn as nn

import torch.nn.functional as F

from .PVTv2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5

from .FusionModule import decoder
from .InteractionModule import InteractionBlock



class PVST(nn.Module):
    """
    PVST (Pyramid Vision Transformer) model for visual data processing.

    This class implements a network architecture that uses a Pyramid Vision Transformer (PVT) as the encoder to extract
    hierarchical features from the input image, followed by interaction blocks and a decoder to process and fuse these
    features for the final output.

    Args:
        args (Namespace): Configuration arguments including model architecture and other hyperparameters.
        embed_dim (int): Embedding dimension for the transformer model.
        dim (int): Dimension for the projection layers.
        img_size (int): Size of the input image.

    Attributes:
        encoder (nn.Module): The PVT-based encoder for feature extraction.
        proj1, proj2, proj3, proj4 (nn.Linear): Projection layers to standardize feature map dimensions.
        interact1, interact2, interact3 (InteractionBlock): Interaction blocks to process and combine features from
            different layers.
        decoder (decoder): A decoder module to fuse features and generate the final output.
    """
    def __init__(self, args, embed_dim=384,dim=96,img_size=224):
        super(PVST, self).__init__()
        # Initialization and configuration
        self.args = args

        self.img_size = img_size
        self.feature_dims = []
        self.dim = dim

        # Initializing the encoder using a specified architecture from the PVTv2 models
        self.encoder = globals()[args.arch]()
        # Loading the pretrained model weights
        pretrained_dict = torch.load('pretrained_model/'+args.arch+'.pth')
        # Filtering out unnecessary keys from the pretrained model
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.encoder.state_dict()}
        # Loading the filtered pretrained weights into the encoder
        self.encoder.load_state_dict(pretrained_dict)

        # Projection layers to transform feature maps to a consistent size
        self.proj1 = nn.Linear(64, 64)
        self.proj2 = nn.Linear(128, 128)
        self.proj3 = nn.Linear(320, 320)
        self.proj4 = nn.Linear(512, 512)

        # Interaction blocks for different layers of features
        self.interact1 = InteractionBlock(dim=320, dim1=512, embed_dim=embed_dim, num_heads=4,
                                          mlp_ratio=3)
        self.interact2 = InteractionBlock(dim=128, dim1=320, dim2=512, embed_dim=embed_dim,
                                          num_heads=2, mlp_ratio=3)
        self.interact3 = InteractionBlock(dim=64, dim1=128, dim2=320, embed_dim=embed_dim,
                                          num_heads=1, mlp_ratio=3)
        # Decoder module to fuse the features and  to generate the final output
        self.decoder = decoder(embed_dim=embed_dim, dims=[64, 128, 320], img_size=img_size, mlp_ratio=1)

    def _freeze_backbone(self, freeze_s1):
        if not freeze_s1:
            return
        assert('resnet' in self.arch and '3x3' not in self.arch)
        m = [self.backbone.conv1, self.backbone.bn1, self.backbone.relu]
        print("freeze stage 0 of resnet")
        for p in m:
            for pp in p.parameters():
                p.requires_grad = False

    def forward(self, image_Input):
        """
        Forward pass of the PVST model.

        Args:
            image_Input (Tensor): The input image tensor.

        Returns:
            Tensor: The output tensor mask after processing the input image through the PVST model.
        """
        B, _, _, _ = image_Input.shape

        # Passing the input through the encoder to get feature maps at different scales
        features = self.encoder(image_Input)

        # Unpacking the feature maps
        fea_1_4, fea_1_8, fea_1_16, fea_1_32 = features

        # Projecting feature maps to a consistent size
        fea_1_4 = self.proj1(fea_1_4)
        fea_1_8 = self.proj2(fea_1_8)
        fea_1_16 = self.proj3(fea_1_16)
        fea_1_32 = self.proj4(fea_1_32)

        # Passing the feature maps through the interaction blocks
        fea_1_16_ = self.interact1(fea_1_16, fea_1_32)
        fea_1_8_ = self.interact2(fea_1_8, fea_1_16_, fea_1_32)
        fea_1_4_ = self.interact3(fea_1_4, fea_1_8_, fea_1_16_)

        # Passing the feature maps through the decoder to generate the final output
        mask = self.decoder([fea_1_16_, fea_1_8_, fea_1_4_])

        return mask