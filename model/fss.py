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
        self.model = RESNET_ARCHS[arch](weights=torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None)

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
        self.image_size = input_size
        self.unet = UNET_ARCHS[unet_arch](n_channels = unet_in_features, n_classes = n_classes, bilinear=bilinear)

        self.fpn = torchvision.ops.FeaturePyramidNetwork([256, 512, 1024, 2048], unet_in_features)

        self.bn_out = nn.BatchNorm2d(unet_in_features)

    def get_relevant(self, fm, m):
        m_c = m.view(m.shape[0],-1)
        m_c = torch.nonzero(m_c, as_tuple=True)
        fm = fm[m_c[0], m_c[1], :]
        return fm

    # def calculate_cos(self, sfm, qfm):
    #     cos = torch.nn.functional.linear(sfm, qfm)
    #     return cos

    def calculate_cos(self, sup_fm, query_fm):
        sup_fm_linear = sup_fm.view(sup_fm.shape[0], sup_fm.shape[1], -1)
        query_fm_linear = query_fm.view(query_fm.shape[0], query_fm.shape[1], -1)

        cos = []
        for i in range(query_fm_linear.shape[0]):
            c = torch.nn.functional.linear(l2_norm(query_fm_linear[i]), l2_norm(sup_fm_linear[i]))
            cos.append(c)
        cos = torch.stack(cos, dim=0)
        return cos

    def forward(self, xq, xs, mask_support):
        # get feature maps
        query_fm1, query_fm2, query_fm3, query_fm4 = self.resnet(xq)
        sup_fm1, sup_fm2, sup_fm3, sup_fm4 = self.resnet(xs)
        batch_size = xq.shape[0]

        fm1_shape = sup_fm1.shape[2:]
        fm2_shape = sup_fm2.shape[2:]
        fm3_shape = sup_fm3.shape[2:]
        fm4_shape = sup_fm4.shape[2:]

        m1 = norm_mask_size(mask_support.type(torch.float), fm1_shape)
        m2 = norm_mask_size(mask_support.type(torch.float), fm2_shape)
        m3 = norm_mask_size(mask_support.type(torch.float), fm3_shape)
        m4 = norm_mask_size(mask_support.type(torch.float), fm4_shape)

        c1 = self.calculate_cos(sup_fm1*m1, query_fm1)
        c2 = self.calculate_cos(sup_fm2*m2, query_fm2)
        c3 = self.calculate_cos(sup_fm3*m3, query_fm3)
        c4 = self.calculate_cos(sup_fm4*m4, query_fm4)

        fpn_input = OrderedDict()

        fpn_input["feat0"] = query_fm1 * torch.nn.functional.softmax(torch.diagonal(c1, dim1=2), dim=-1).unsqueeze(2).unsqueeze(3)
        fpn_input["feat1"] = query_fm2 * torch.nn.functional.softmax(torch.diagonal(c2, dim1=2), dim=-1).unsqueeze(2).unsqueeze(3)
        fpn_input["feat2"] = query_fm3 * torch.nn.functional.softmax(torch.diagonal(c3, dim1=2), dim=-1).unsqueeze(2).unsqueeze(3)
        fpn_input["feat3"] = query_fm4 * torch.nn.functional.softmax(torch.diagonal(c4, dim1=2), dim=-1).unsqueeze(2).unsqueeze(3)

        unet_input = self.fpn(fpn_input)
        unet_input = self.prepare_unet_input(unet_input)
        unet_input = self.bn_out(unet_input)

        out = self.unet(unet_input)
        return out

    def prepare_unet_input(self, unet_input):
        out = torch.nn.functional.interpolate(unet_input["feat0"], scale_factor=2, mode='nearest')
        out += torch.nn.functional.interpolate(unet_input["feat1"], scale_factor=4, mode='nearest')
        out += torch.nn.functional.interpolate(unet_input["feat2"], scale_factor=8, mode='nearest')
        out += torch.nn.functional.interpolate(unet_input["feat3"], scale_factor=16, mode='nearest')
        return out

    # def forward(self, xq, xs, mask_support):
    #     # get feature maps
    #     query_fm1, query_fm2, query_fm3, query_fm4 = self.resnet(xq)
    #     sup_fm1, sup_fm2, sup_fm3, sup_fm4 = self.resnet(xs)
    #     batch_size = xq.shape[0]
    #
    #     fm1_shape = sup_fm1.shape[2:]
    #     fm2_shape = sup_fm2.shape[2:]
    #     fm3_shape = sup_fm3.shape[2:]
    #     fm4_shape = sup_fm4.shape[2:]
    #
    #     m1 = norm_mask_size(mask_support.type(torch.float), fm1_shape)
    #     m2 = norm_mask_size(mask_support.type(torch.float), fm2_shape)
    #     m3 = norm_mask_size(mask_support.type(torch.float), fm3_shape)
    #     m4 = norm_mask_size(mask_support.type(torch.float), fm4_shape)
    #
    #     query_fm1 = l2_norm(reshape_feat_map(query_fm1))
    #     query_fm2 = l2_norm(reshape_feat_map(query_fm2))
    #     query_fm3 = l2_norm(reshape_feat_map(query_fm3))
    #     query_fm4 = l2_norm(reshape_feat_map(query_fm4))
    #
    #     sup_fm1 = l2_norm(reshape_feat_map(sup_fm1))
    #     sup_fm2 = l2_norm(reshape_feat_map(sup_fm2))
    #     sup_fm3 = l2_norm(reshape_feat_map(sup_fm3))
    #     sup_fm4 = l2_norm(reshape_feat_map(sup_fm4))
    #
    #     # query_fm1 = self.get_relevant(query_fm1, m1)
    #     # query_fm2 = self.get_relevant(query_fm2, m2)
    #     # query_fm3 = self.get_relevant(query_fm3, m3)
    #     # query_fm4 = self.get_relevant(query_fm4, m4)
    #     #
    #     # sup_fm1 = self.get_relevant(sup_fm1, m1)
    #     # sup_fm2 = self.get_relevant(sup_fm2, m2)
    #     # sup_fm3 = self.get_relevant(sup_fm3, m3)
    #     # sup_fm4 = self.get_relevant(sup_fm4, m4)
    #     #
    #     # cos1 = self.calculate_cos(query_fm1, sup_fm1)
    #     # cos2 = self.calculate_cos(query_fm2, sup_fm2)
    #     # cos3 = self.calculate_cos(query_fm3, sup_fm3)
    #     # cos4 = self.calculate_cos(query_fm4, sup_fm4)
    #
    #     unet_inputs = []
    #     for bs in range(batch_size):
    #         cos1 = torch.nn.functional.linear(query_fm1[bs], sup_fm1[bs])
    #         cos2 = torch.nn.functional.linear(query_fm2[bs], sup_fm2[bs])
    #         cos3 = torch.nn.functional.linear(query_fm3[bs], sup_fm3[bs])
    #         cos4 = torch.nn.functional.linear(query_fm4[bs], sup_fm4[bs])
    #
    #         # input: batch_size, N^2, N^2 | batch_size*N^2, batch_size*N^2 ->
    #         # output: batch_size, N^2, N, N
    #         cos1 = cos1.reshape(fm1_shape[0]*fm1_shape[1], fm1_shape[0], fm1_shape[1])
    #         cos2 = cos2.reshape(fm2_shape[0]*fm2_shape[1], fm2_shape[0], fm2_shape[1])
    #         cos3 = cos3.reshape(fm3_shape[0]*fm3_shape[1], fm3_shape[0], fm3_shape[1])
    #         cos4 = cos4.reshape(fm4_shape[0]*fm4_shape[1], fm4_shape[0], fm4_shape[1])
    #
    #         # 2, 1, 64, 64
    #         # m1flat = m1[bs].reshape(m1[bs].shape[0], m1[bs].shape[1]*m1[bs].shape[2])
    #         # cos1   = cos1 * m1flat.squeeze(0).unsqueeze(1).unsqueeze(2) # 1, 64*64, 64, 64 # 1, 64*64
    #         # m2flat = m2[bs].reshape(m2[bs].shape[0], m2[bs].shape[1] * m2[bs].shape[2])
    #         # cos2   = cos2* m2flat.squeeze(0).unsqueeze(1).unsqueeze(2)
    #         # m3flat = m3[bs].reshape(m3[bs].shape[0], m3[bs].shape[1] * m3[bs].shape[2])
    #         # cos3   = cos3 * m3flat.squeeze(0).unsqueeze(1).unsqueeze(2)
    #         # m4flat = m4[bs].reshape(m4[bs].shape[0], m4[bs].shape[1] * m4[bs].shape[2])
    #         # cos4   = cos4 * m4flat.squeeze(0).unsqueeze(1).unsqueeze(2)
    #         # del m1flat, m2flat, m3flat, m4flat; torch.cuda.empty_cache()
    #
    #         cos1 = norm_feature_map_size(cos1.unsqueeze(0), target_size=(self.image_size, self.image_size))
    #         cos2 = norm_feature_map_size(cos2.unsqueeze(0), target_size=(self.image_size, self.image_size))
    #         cos3 = norm_feature_map_size(cos3.unsqueeze(0), target_size=(self.image_size, self.image_size))
    #         cos4 = norm_feature_map_size(cos4.unsqueeze(0), target_size=(self.image_size, self.image_size))
    #
    #         cos1 = self.feat_map_reduction_1(cos1)
    #         cos2 = self.feat_map_reduction_2(cos2)
    #         cos3 = self.feat_map_reduction_3(cos3)
    #         cos4 = self.feat_map_reduction_4(cos4)
    #
    #         unet_inputs.append(torch.cat([cos1.squeeze(0), cos2.squeeze(0), cos3.squeeze(0), cos4.squeeze(0)], dim=0))
    #
    #     # del sup_fm1, sup_fm2, sup_fm3, sup_fm4, query_fm1, query_fm2, query_fm4, query_fm4; torch.cuda.empty_cache()
    #
    #     unet_inputs = torch.stack(unet_inputs, dim=0)
    #     unet_inputs = self.bn_out(unet_inputs)
    #     unet_inputs = self.relu_out(unet_inputs)
    #
    #     out = self.unet(unet_inputs)
    #     return out
    
    
    
    
    
    
    
    