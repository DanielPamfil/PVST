import torch
import torch.nn as nn

import torch.nn.functional as F

from .PVTv2 import pvt_v2_b4, pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3

from .FusionModule import decoder
from .InteractionModule import InteractionBlock



class PVST(nn.Module):
    def __init__(self, args, embed_dim=384,dim=96,img_size=224):
        super(PVST, self).__init__()

        self.args = args

        self.img_size = img_size
        self.feature_dims = []
        self.dim = dim

        self.encoder = globals()[args.arch]()

        pretrained_dict = torch.load('pretrained_model/'+args.arch+'.pth')

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(pretrained_dict)

        self.proj1 = nn.Linear(64, 64)
        self.proj2 = nn.Linear(128, 128)
        self.proj3 = nn.Linear(320, 320)
        self.proj4 = nn.Linear(512, 512)

        self.interact1 = InteractionBlock(dim=320, dim1=512, embed_dim=embed_dim, num_heads=4,
                                          mlp_ratio=3)
        self.interact2 = InteractionBlock(dim=128, dim1=320, dim2=512, embed_dim=embed_dim,
                                          num_heads=2, mlp_ratio=3)
        self.interact3 = InteractionBlock(dim=64, dim1=128, dim2=320, embed_dim=embed_dim,
                                          num_heads=1, mlp_ratio=3)

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
        B, _, _, _ = image_Input.shape

        features = self.encoder(image_Input)

        fea_1_4, fea_1_8, fea_1_16, fea_1_32 = features
        fea_1_4 = self.proj1(fea_1_4)
        fea_1_8 = self.proj2(fea_1_8)
        fea_1_16 = self.proj3(fea_1_16)
        fea_1_32 = self.proj4(fea_1_32)
        fea_1_16_ = self.interact1(fea_1_16, fea_1_32)
        fea_1_8_ = self.interact2(fea_1_8, fea_1_16_, fea_1_32)
        fea_1_4_ = self.interact3(fea_1_4, fea_1_8_, fea_1_16_)

        mask = self.decoder([fea_1_16_, fea_1_8_, fea_1_4_])

        return mask