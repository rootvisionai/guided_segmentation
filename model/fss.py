# full assembly of the sub-parts to form the complete net

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.utils

from .unet_parts import down, up, outconv, inconv, double_conv
from .utils import l2_norm, reshape_feat_map, norm_feature_map_size, norm_mask_size, convert_3d_2d

from collections import OrderedDict

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,bilinear=True):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256,bilinear)
        self.up2 = up(512, 128,bilinear)
        self.up3 = up(256, 64,bilinear)
        self.up4 = up(128, 64,bilinear)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class FSUNet(nn.Module):
    def __init__(self, n_channels, n_classes,bilinear=True):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64) # 512->512
        self.down1 = down(64, 128)        # 512->256
        self.down2 = down(128, 256)       # 256->128
        self.down3 = down(256, 512)       # 128->64
        self.down4 = down(512, 512)       # 128->64
        self.up1 = up(1024, 256,bilinear)
        self.up2 = up(512, 128,bilinear)
        self.up3 = up(256, 64,bilinear)
        self.up4 = up(128, 64,bilinear)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes,bilinear=True):
        super(UNet2, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        
        self.P4 = double_conv(512,512)
        self.P3 = double_conv(256,256)
        self.P2 = double_conv(128,128)
        self.P1 = double_conv(64,64)
        
        self.up1 = up(1024, 256,bilinear)
        self.up2 = up(512, 128,bilinear)
        self.up3 = up(256, 64,bilinear)
        self.up4 = up(128, 64,bilinear)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, self.P4(x4))
        x = self.up2(x, self.P3(x3))
        x = self.up3(x, self.P2(x2))
        x = self.up4(x, self.P1(x1))
        x = self.outc(x)
        return x


class UNet3(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet3, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        
        self.P4 = double_conv(512,512)
        self.P3 = double_conv(256,256)
        self.P2 = double_conv(128,128)
        self.P1 = double_conv(64,64)
        
        self.P6 = double_conv(512,512)
        self.P5 = double_conv(256,256)
        
        self.up1 = up(1024, 256, bilinear)
        self.up2 = up(512, 128, bilinear)
        self.up3 = up(256, 64, bilinear)
        self.up4 = up(128, 64, bilinear)
        self.up4 = up(128, 64, bilinear)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, self.P6(self.P4(x4)))
        x = self.up2(x, self.P5(self.P3(x3)))
        x = self.up3(x, self.P2(x2))
        x = self.up4(x, self.P1(x1))
        x = self.outc(x)
        return x


from torchvision.models import resnet18
from torchvision.models import resnet34
from torchvision.models import resnet50
from torchvision.models import resnet101

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
        self.unet = UNET_ARCHS[unet_arch](n_channels = unet_in_features, n_classes = n_classes, bilinear=bilinear)

        self.fpn = torchvision.ops.FeaturePyramidNetwork([256, 512, 1024, 2048], int(unet_in_features/4))

        self.bn_out = nn.BatchNorm2d(unet_in_features)

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
            query_fm1, query_fm2, query_fm3, query_fm4 = self.resnet(xq)
            sup_fm1, sup_fm2, sup_fm3, sup_fm4 = self.resnet(xs)

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
        unet_input = self.prepare_unet_input(unet_input)
        unet_input = self.bn_out(unet_input)

        out = self.unet(unet_input)
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
        outs_negative = self.one_way_k_shot(xq, [1-xs_ for xs_ in xs], [1-ms_ for ms_ in ms])
        outs = torch.cat([outs_negative, outs_positive], dim=1)
        return outs

    def infer(self, xq, xs, ms, duplicate=True):
        with torch.no_grad():
            pred = self.forward(xq, xs, ms)
        if duplicate:
            pred = torch.nn.functional.interpolate(pred, size=(xq.shape[2], xq.shape[3]))
        return pred

    def prepare_unet_input(self, unet_input):
        # out = self.mlu([
        #     unet_input["feat0"],
        #     unet_input["feat1"],
        #     unet_input["feat2"],
        #     unet_input["feat3"]
        # ])

        # out = torch.nn.functional.interpolate(unet_input["feat0"], scale_factor=2, mode='nearest')
        # out += torch.nn.functional.interpolate(unet_input["feat1"], scale_factor=4, mode='nearest')
        # out += torch.nn.functional.interpolate(unet_input["feat2"], scale_factor=8, mode='nearest')
        # out += torch.nn.functional.interpolate(unet_input["feat3"], scale_factor=16, mode='nearest')

        out = torch.cat([
            torch.nn.functional.interpolate(unet_input["feat0"], scale_factor=4, mode='nearest'),
            torch.nn.functional.interpolate(unet_input["feat1"], scale_factor=8, mode='nearest'),
            torch.nn.functional.interpolate(unet_input["feat2"], scale_factor=16, mode='nearest'),
            torch.nn.functional.interpolate(unet_input["feat3"], scale_factor=32, mode='nearest')
        ], dim=1)
        return out
