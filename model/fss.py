import torch
import torch.nn as nn
import torchvision.utils
from torchvision.models import resnet18
from torchvision.models import resnet34
from torchvision.models import resnet50
from torchvision.models import resnet101

from .utils import l2_norm, norm_mask_size
from .unet import UNet, UNet2, UNet3

from collections import OrderedDict



RESNET_ARCHS = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101
}

RESNET_WEIGHTS = {
    "resnet18": torchvision.models.ResNet18_Weights.DEFAULT,
    "resnet34": torchvision.models.ResNet34_Weights.DEFAULT,
    "resnet50": torchvision.models.ResNet50_Weights.DEFAULT,
    "resnet101": torchvision.models.ResNet101_Weights.DEFAULT
}

class ResNet(nn.Module):
    def __init__(self, arch, pretrained=True):
        """
        Initialize with (architecture: str, pretrained: bool)
        
        forward()
            Input: torch.tensor 
                with shape:
                    batch_size   (B)
                    channel      (C)
                    height/row   (H)
                    width/column (W)
                                            
            Output: (torch.tensor,torch.tensor,torch.tensor,torch.tensor)
                with shapes:
                    batch_size   (B, B, B, B)
                    channel      (64, 128, 256, 512)
                    height/row   (H/4, H/8, H/16, H/32)
                    width/column (W/4, W/8, W/16, W/32)
        """
        super(ResNet, self).__init__()
        self.model = RESNET_ARCHS[arch](weights=RESNET_WEIGHTS[arch] if pretrained else None)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        fm1 = self.model.layer1(x)

        fm2 = self.model.layer2(fm1)

        fm3 = self.model.layer3(fm2)

        fm4 = self.model.layer4(fm3)
        
        return (fm1, fm2, fm3, fm4)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResidualUpsample(nn.Module):
    def __init__(self, in_channel=64):
        super().__init__()

        self.residual1 = ResidualBlock(in_channels=in_channel, out_channels=in_channel, stride=1, downsample=None)
        self.residual2 = ResidualBlock(in_channels=in_channel, out_channels=in_channel, stride=1, downsample=None)
        self.residual3 = ResidualBlock(in_channels=in_channel, out_channels=in_channel, stride=1, downsample=None)
        self.residual4 = ResidualBlock(in_channels=in_channel, out_channels=in_channel, stride=1, downsample=None)
        self.outc = nn.Conv2d(in_channel, 1, 1)

    def forward(self, unet_input):
        x = torch.nn.functional.interpolate(unet_input["feat3"], scale_factor=2, mode='bilinear')
        x = self.residual1(x) + unet_input["feat2"]
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.residual2(x) + unet_input["feat1"]
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.residual3(x) + unet_input["feat0"]
        x = torch.nn.functional.interpolate(x, scale_factor=4, mode='bilinear')
        x = self.residual4(x)
        x = self.outc(x)
        return x

class CorrFC(nn.Module):
    def __init__(self, feat_sizes):
        super().__init__()

        self.fc0 = nn.Sequential(
            nn.Linear(feat_sizes[0], feat_sizes[0]),
            nn.Softmax(dim=-1)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(feat_sizes[1], feat_sizes[1]),
            nn.Softmax(dim=-1)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(feat_sizes[2], feat_sizes[2]),
            nn.Softmax(dim=-1)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(feat_sizes[3], feat_sizes[3]),
            nn.Softmax(dim=-1)
        )

    def forward(self, feats):
        out0 = self.fc0(feats[0])
        out1 = self.fc1(feats[1])
        out2 = self.fc2(feats[2])
        out3 = self.fc3(feats[3])
        return out0, out1, out2, out3

UNET_ARCHS = {
    "unet1": UNet,
    "unet2": UNet2,
    "unet3": UNet3
}
class FSS(nn.Module):
    def __init__(
            self, 
            input_size = 512,
            resnet_arch = "resnet18", 
            unet_arch="unet3", 
            n_classes=1,
            unet_in_features=256,
            bilinear=True
        ):
        
        super(FSS, self).__init__()
        self.resnet = ResNet(resnet_arch, pretrained=True)
        self.corrfc = CorrFC([256, 512, 1024, 2048])
        self.image_size = input_size
        # self.unet = UNET_ARCHS[unet_arch](n_channels = unet_in_features, n_classes = n_classes, bilinear=bilinear)

        self.fpn = torchvision.ops.FeaturePyramidNetwork([256, 512, 1024, 2048], unet_in_features)
        self.residual_up = ResidualUpsample(in_channel=unet_in_features)


    def get_relevant(self, fm, m):
        m_c = m.view(m.shape[0],-1)
        m_c = torch.nonzero(m_c, as_tuple=True)
        fm = fm[m_c[0], m_c[1], :]
        return fm

    def calculate_cos(self, sup_fm, query_fm):
        sup_fm_linear = sup_fm.view(sup_fm.shape[0], sup_fm.shape[1], -1)
        query_fm_linear = query_fm.view(query_fm.shape[0], query_fm.shape[1], -1)

        cos = []
        for i in range(query_fm_linear.shape[0]):
            c = torch.nn.functional.linear(l2_norm(query_fm_linear[i]), l2_norm(sup_fm_linear[i]))
            cos.append(c)
        cos = torch.stack(cos, dim=0)
        return cos

    def one_way_one_shot(self, xq, xs, ms):
        """
        xq: query image
        xs: support image
        ms: support mask
        """

        # get feature maps
        with torch.no_grad():
            sup_fm1, sup_fm2, sup_fm3, sup_fm4 = self.resnet(xs)

        query_fm1, query_fm2, query_fm3, query_fm4 = self.resnet(xq)

        fm1_shape = sup_fm1.shape[2:]
        fm2_shape = sup_fm2.shape[2:]
        fm3_shape = sup_fm3.shape[2:]
        fm4_shape = sup_fm4.shape[2:]

        m1 = norm_mask_size(ms.type(torch.float), fm1_shape)
        m2 = norm_mask_size(ms.type(torch.float), fm2_shape)
        m3 = norm_mask_size(ms.type(torch.float), fm3_shape)
        m4 = norm_mask_size(ms.type(torch.float), fm4_shape)

        c1 = self.calculate_cos(sup_fm1*m1, query_fm1)
        c2 = self.calculate_cos(sup_fm2*m2, query_fm2)
        c3 = self.calculate_cos(sup_fm3*m3, query_fm3)
        c4 = self.calculate_cos(sup_fm4*m4, query_fm4)

        fpn_input = OrderedDict()

        c1 = torch.diagonal(c1, dim1=2)
        c2 = torch.diagonal(c2, dim1=2)
        c3 = torch.diagonal(c3, dim1=2)
        c4 = torch.diagonal(c4, dim1=2)

        c1, c2, c3, c4 = self.corrfc([c1, c2, c3, c4])

        fpn_input["feat0"] = query_fm1 * c1.unsqueeze(2).unsqueeze(3)
        fpn_input["feat1"] = query_fm2 * c2.unsqueeze(2).unsqueeze(3)
        fpn_input["feat2"] = query_fm3 * c3.unsqueeze(2).unsqueeze(3)
        fpn_input["feat3"] = query_fm4 * c4.unsqueeze(2).unsqueeze(3)

        unet_input = self.fpn(fpn_input)
        out = self.residual_up(unet_input)
        return out

    def one_way_k_shot(self, xq, xs, ms):
        """
        xq: query image
        xs: support image
        ms: support mask
        """

        for i in range(len(ms)): # index N shot
            if i == 0:
                outs = self.one_way_one_shot(xq, xs[i], ms[i])
            else:
                outs += self.one_way_one_shot(xq, xs[i], ms[i])

        return outs

    def forward(self, xq, xs, ms):
        # positive
        outs_positive = self.one_way_k_shot(xq, xs, ms)
        # outs_negative = self.one_way_k_shot(xq, [1-xs_ for xs_ in xs], [1-ms_ for ms_ in ms])
        # outs = torch.cat([outs_negative, outs_positive], dim=1)
        return outs_positive

    def infer(self, xq, xs, ms, duplicate=True):
        with torch.no_grad():
            pred = self.forward(xq, xs, ms)
        if duplicate:
            pred = torch.nn.functional.interpolate(pred, size=(xq.shape[2], xq.shape[3]))
        return pred

    def prepare_unet_input(self, unet_input):
        out = torch.cat([
            torch.nn.functional.interpolate(unet_input["feat0"], scale_factor=4, mode='bipolar'),
            torch.nn.functional.interpolate(unet_input["feat1"], scale_factor=8, mode='bipolar'),
            torch.nn.functional.interpolate(unet_input["feat2"], scale_factor=16, mode='bipolar'),
            torch.nn.functional.interpolate(unet_input["feat3"], scale_factor=32, mode='bipolar')
        ], dim=1)
        return out
