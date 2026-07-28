import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import resnet50
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import FeaturePyramidNetwork


class TemporalAttention(nn.Module):
    """
    Temporal attention module.

    Input:
        x: (B,T,C,H,W)

    Output:
        fused: (B,C,H,W)
    """

    def __init__(self, channels):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Conv3d(channels,channels // 2,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 2, 1,kernel_size=1)
        )

    def forward(self, x):
        # x:
        # B,T,C,H,W

        # Conv3d expects:
        # B,C,T,H,W
        x_perm = x.permute(0,2,1,3,4)
        weights = self.attention(x_perm)

        # B,1,T,H,W
        weights = torch.softmax(weights,dim=2)
        weighted = x_perm * weights
        fused = weighted.sum(dim=2)

        # B,C,H,W
        return fused

class TemporalResNet50(nn.Module):

    def __init__(
        self,
        input_dim=1,
        t_frames=5
        ):

        super().__init__()

        backbone = resnet50(weights=None, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
        backbone.conv1 = nn.Conv2d(input_dim,64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Feature extraction only
        self.body = nn.Module()
        self.body.conv1 = backbone.conv1
        self.body.bn1 = backbone.bn1
        self.body.relu = backbone.relu
        self.body.maxpool = backbone.maxpool
        self.body.layer1 = backbone.layer1
        self.body.layer2 = backbone.layer2
        self.body.layer3 = backbone.layer3
        self.body.layer4 = backbone.layer4

        # Temporal attention modules
        self.temporal1 = TemporalAttention(256)
        self.temporal2 = TemporalAttention(512)
        self.temporal3 = TemporalAttention(1024)
        self.temporal4 = TemporalAttention(2048)

        # Feed the weighted representations into FPN
        self.fpn = FeaturePyramidNetwork(in_channels_list=[256,512,1024,2048],out_channels=256)
        # Number of output channels for backbone
        # FasterRCNN requires this
        self.out_channels = 256

    def extract_frame_features(self, x):
        """
        Extract ResNet features for one frame.

        Input:
            x: (B,1,H,W)

        Output:
            feature maps
        """
        x = self.body.conv1(x)
        x = self.body.bn1(x)
        x = self.body.relu(x)
        x = self.body.maxpool(x)

        c1 = self.body.layer1(x)
        c2 = self.body.layer2(c1)
        c3 = self.body.layer3(c2)
        c4 = self.body.layer4(c3)

        return c1,c2,c3,c4

    def forward(self, images):
        print("BACKBONE INPUT:", images.shape) # TESTING (!)
        
        features = {"0":[],"1":[],"2":[],"3":[]}
        
        for sequence in images: # (T,1,H,W)
            T, C, H, W = sequence.shape
            # Add batch dimension
            sequence = sequence.unsqueeze(0) # (1,T,1,H,W)
            B = sequence.shape[0]
            # Merge batch and temporal dimensions
            sequence = sequence.reshape(B*T,C,H,W)

            c1, c2, c3, c4 = self.extract_frame_features(sequence)
            
            # Restore temporal dimension
            c1 = c1.reshape(B,T, c1.shape[1], c1.shape[2],c1.shape[3])
            c2 = c2.reshape(B,T, c2.shape[1], c2.shape[2],c2.shape[3])
            c3 = c3.reshape(B,T,c3.shape[1],c3.shape[2],c3.shape[3])
            c4 = c4.reshape(B,T,c4.shape[1],c4.shape[2],c4.shape[3])

            # Temporal attention
            # (B,C,H,W)
            features["0"].append(self.temporal1(c1))
            features["1"].append(self.temporal2(c2))
            features["2"].append(self.temporal3(c3))
            features["3"].append(self.temporal4(c4))

        # Combine batch samples
        features = OrderedDict({
            "0": torch.cat(features["0"], dim=0),
            "1": torch.cat(features["1"], dim=0),
            "2": torch.cat(features["2"], dim=0),
            "3": torch.cat(features["3"], dim=0)})

        # FPN
        features = self.fpn(features)

        return features