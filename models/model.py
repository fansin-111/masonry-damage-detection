import torch
import torch.nn as nn

from .modules import Conv1x1, FeatureExtractionModule, PointFeaturePropagation
from pointnet2.pointnet2_modules import PointnetSAModule

class Model(nn.Module):
    """Final improved PointNet++ model with the complete NSA module."""

    def __init__(self, args, num_classes=2):
        super().__init__()

        init_channel = 16

        radius = args.radius

        self.le0 = Conv1x1(9, init_channel)

        self.le1 = PointnetSAModule(
                    npoint=args.num_points,
                    radius=0.05,
                    nsample=20,
                    mlp=[init_channel, init_channel, init_channel],
                    use_xyz=True,
                    bn=True
        )

       # encoder
        self.conv1 = FeatureExtractionModule(init_channel, init_channel*2, radius)

        self.conv2 = FeatureExtractionModule(init_channel*2, init_channel*4, radius*2)

        self.conv3 = FeatureExtractionModule(init_channel*4, init_channel*8, radius*4)

        self.conv4 = FeatureExtractionModule(init_channel*8, init_channel*16, radius*8)

        self.fp3 = PointFeaturePropagation(in_channel1=init_channel*16, in_channel2=init_channel*8, out_channel=init_channel*8)
        self.up_conv4 = FeatureExtractionModule(init_channel*8, init_channel*8, radius*4)

        self.fp2 = PointFeaturePropagation(in_channel1=init_channel*8, in_channel2=init_channel*4, out_channel=init_channel*8)
        self.up_conv3 = FeatureExtractionModule(init_channel*8, init_channel*8, radius*2)

        self.fp1 = PointFeaturePropagation(in_channel1=init_channel*8, in_channel2=init_channel*2, out_channel=init_channel*8)
        self.up_conv2 = FeatureExtractionModule(init_channel*8, init_channel*8, radius)

        self.fp0 = PointFeaturePropagation(in_channel1=init_channel*8, in_channel2=16, out_channel=init_channel*8)
        self.up_conv1 = nn.Sequential(Conv1x1(init_channel*8+3, 256),
                                      nn.Dropout(args.dropout),
                                      Conv1x1(256, 128),
                                      Conv1x1(128, 128))
        
        self.conv = nn.Linear(128, num_classes, bias=False)
               
    def forward(self, x):
        x = x.permute(0,2,1)
        xyz = x[:,:,:3].clone().contiguous()
        point_num = xyz.size(1)
        x = self.le0(x)

        x = x.transpose(1,2).contiguous()
        l1_xyz, l1_feat = self.le1(xyz, x) # [B, N, 3] -> [B, N, 16]
        l1_feat = l1_feat.transpose(1,2).contiguous()   
        x = x.transpose(1,2).contiguous()
              
        # encoder
        l1_feat, l1_xyz = self.conv1(l1_feat, l1_xyz, point_num//2) 
        l2_feat, l2_xyz = self.conv2(l1_feat, l1_xyz, point_num//4) 
        l3_feat, l3_xyz = self.conv3(l2_feat, l2_xyz, point_num//8) 
        l4_feat, l4_xyz = self.conv4(l3_feat, l3_xyz, point_num//16) 

        l3_feat = self.fp3(l3_xyz, l4_xyz, l3_feat, l4_feat) 
        l3_feat, l3_xyz = self.up_conv4(l3_feat, l3_xyz, point_num//8) 

        l2_feat = self.fp2(l2_xyz, l3_xyz, l2_feat, l3_feat) 
        l2_feat, l2_xyz = self.up_conv3(l2_feat, l2_xyz, point_num//4) 

        l1_feat = self.fp1(l1_xyz, l2_xyz, l1_feat, l2_feat)
        l1_feat, l1_xyz = self.up_conv2(l1_feat, l1_xyz, point_num//2)    

        # feature fusion
        l1_feat = self.fp0(xyz, l1_xyz, x, l1_feat)
        feat = torch.cat((xyz, l1_feat), dim=-1) 
        feat = self.up_conv1(feat)
        feat = self.conv(feat)
        feat = feat.transpose(1,2)
        return feat
    
