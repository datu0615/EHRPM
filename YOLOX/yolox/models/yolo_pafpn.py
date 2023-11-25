#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv, DilateConv

# class YOLOPAFPN(nn.Module):
#     """
#     YOLOv3 model. Darknet 53 is the default backbone of this model.
#     """

#     def __init__(
#         self,
#         depth=1.0,
#         width=1.0,
#         in_features=("dark2", "dark3", "dark4", "dark5"),
#         in_channels=[256, 512, 1024],
#         depthwise=False,
#         act="silu",
#     ):
#         super().__init__()
#         self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
#         self.in_features = in_features
#         self.in_channels = in_channels
#         Conv = DWConv if depthwise else BaseConv

#         self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
#         self.lateral_conv0 = BaseConv(
#             int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
#         )
        
        
#         self.downsample = Conv(
#             1, 1, 3, 2, act=act
#         )
        
#         self.sigmoid = nn.Sigmoid()
        
        
#         self.C3_p4 = CSPLayer(
#             int(2 * in_channels[1] * width)+48,
#             int(in_channels[1] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )  # cat

#         self.reduce_conv1 = BaseConv(
#             int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
#         )
#         self.C3_p3 = CSPLayer(
#             int(2 * in_channels[0] * width)+48,
#             int(in_channels[0] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         # bottom-up conv
#         self.bu_conv2 = Conv(
#             int(in_channels[0] * width), int(in_channels[0] * (width) / 2), 3, 2, act=act
#         )
#         self.bu_dlconv2 = DilateConv(
#             int(in_channels[0] * width), int(in_channels[0] * (width) / 2), 3, 2, act=act
#         )
#         self.C3_n3 = CSPLayer(
#             int(2 * in_channels[0] * width),
#             int(in_channels[1] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         # bottom-up conv
#         self.bu_conv1 = Conv(
#             int(in_channels[1] * width), int(in_channels[1] * (width) / 2), 3, 2, act=act
#         )
#         self.bu_dlconv1 = DilateConv(
#             int(in_channels[1] * width), int(in_channels[1] * (width) / 2), 3, 2, act=act
#         )
#         self.C3_n4 = CSPLayer(
#             int(2 * in_channels[1] * width),
#             int(in_channels[2] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         self.conv1x1_bu2 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
#         self.conv1x1_bu1 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
#         self.conv1x1_bu0 = BaseConv(int(in_channels[2] * width)*2, int(in_channels[2] * width), 1, 1, act=act)

#         self.conv1x1_2 = BaseConv(int(in_channels[0] * width), 1, 1, 1, act=act)
#         self.conv1x1_1 = BaseConv(int(in_channels[1] * width), 1, 1, 1, act=act)
#         self.conv1x1_0 = BaseConv(int(in_channels[2] * width), 1, 1, 1, act=act)
    

#     def forward(self, input):
#         """
#         Args:
#             inputs: input images.

#         Returns:
#             Tuple[Tensor]: FPN feature.
#         """
#         #DICONV_featuremap = Before_Module.DICONV
#         #print(DICONV_featuremap.shape)
#         #  backbone
#         out_features = self.backbone(input)
#         features = [out_features[f] for f in self.in_features]
#         [x3, x2, x1, x0] = features
#         #print(x2.shape, x1.shape, x0.shape)
        
#         fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
#         f_out0 = self.upsample(fpn_out0)  # 512/16
#         f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
#         b,c,w,h = f_out0.shape
#         x3_resize2 = F.interpolate(x3, size=(w,h), mode='bilinear', align_corners=False)
#         f_out0 = torch.cat([f_out0, x3_resize2], 1)
#         f_out0 = self.C3_p4(f_out0)  # 1024->512/16

#         fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
#         f_out1 = self.upsample(fpn_out1)  # 256/8
#         f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
#         b,c,w,h = f_out1.shape
#         x3_resize1 = F.interpolate(x3, size=(w,h), mode='bilinear', align_corners=False)
#         f_out1 = torch.cat([f_out1, x3_resize1], 1)
#         pan_out2 = self.C3_p3(f_out1)  # 512->256/8
#         pan_out2 = torch.cat([pan_out2, x2], 1)
#         pan_out2 = self.conv1x1_bu2(pan_out2)

#         p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
#         p_dlout1 = self.bu_dlconv2(pan_out2)  # 256->256/16
#         p_out1 = torch.cat([p_out1, p_dlout1], 1)
#         p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
#         pan_out1 = self.C3_n3(p_out1)  # 512->512/16
#         pan_out1 = torch.cat([pan_out1, x1], 1)
#         pan_out1 = self.conv1x1_bu1(pan_out1)

#         p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
#         p_dlout0 = self.bu_dlconv1(pan_out1)  # 512->512/32
#         p_out0 = torch.cat([p_out0, p_dlout0], 1)
#         p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
#         pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
#         pan_out0 = torch.cat([pan_out0, x0], 1)
#         pan_out0 = self.conv1x1_bu0(pan_out0)
        
        
#         ### after module creation
#         pan_out2_p3 = self.conv1x1_2(pan_out2)
#         pan_out1_n3 = self.conv1x1_1(pan_out1)
#         pan_out0_n4 = self.conv1x1_0(pan_out0)        
        
#         # pan_out2_p3 = torch.mean(pan_out2, dim=1, keepdim = True)
#         # pan_out1_n3 = torch.mean(pan_out1, dim=1, keepdim = True)
#         # pan_out0_n4 = torch.mean(pan_out0, dim=1, keepdim = True)
        
#         pan_out2_p3 = self.sigmoid(pan_out2_p3)
#         pan_out1_n3 = self.sigmoid(pan_out1_n3)
#         pan_out0_n4 = self.sigmoid(pan_out0_n4)
#         #print(pan_out2_p3.shape, pan_out1_n3.shape, pan_out0_n4.shape)
        
#         pan_out2 = pan_out2 * self.upsample(pan_out1_n3)
#         pan_out1 = pan_out1 * self.upsample(pan_out0_n4)
#         pan_out0 = pan_out0 * self.downsample(self.downsample(pan_out2_p3))
    
#         outputs = (pan_out2, pan_out1, pan_out0)
#         #print(outputs[0].shape, outputs[1].shape, outputs[2].shape)
#         return outputs




# class YOLOPAFPN(nn.Module):
#     """
#     YOLOv3 model. Darknet 53 is the default backbone of this model.
#     """

#     def __init__(
#         self,
#         depth=1.0,
#         width=1.0,
#         in_features=("dark3", "dark4", "dark5"),
#         in_channels=[256, 512, 1024],
#         depthwise=False,
#         act="silu",
#     ):
#         super().__init__()
#         self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
#         self.in_features = in_features
#         self.in_channels = in_channels
#         Conv = DWConv if depthwise else BaseConv

#         self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
#         self.lateral_conv0 = BaseConv(
#             int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
#         )
#         self.C3_p4 = CSPLayer(
#             int(2 * in_channels[1] * width),
#             int(in_channels[1] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )  # cat

#         self.reduce_conv1 = BaseConv(
#             int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
#         )
#         self.C3_p3 = CSPLayer(
#             int(2 * in_channels[0] * width),
#             int(in_channels[0] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         # bottom-up conv
#         self.bu_conv2 = Conv(
#             int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
#         )
#         self.C3_n3 = CSPLayer(
#             int(2 * in_channels[0] * width),
#             int(in_channels[1] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         # bottom-up conv
#         self.bu_conv1 = Conv(
#             int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
#         )
#         self.C3_n4 = CSPLayer(
#             int(2 * in_channels[1] * width),
#             int(in_channels[2] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         self.downsample = Conv(
#             1, 1, 3, 2, act=act
#         )
        
#         self.sigmoid = nn.Sigmoid()

#         self.conv1x1_2 = BaseConv(int(in_channels[0] * width), 1, 1, 1, act=act)
#         self.conv1x1_1 = BaseConv(int(in_channels[1] * width), 1, 1, 1, act=act)
#         self.conv1x1_0 = BaseConv(int(in_channels[2] * width), 1, 1, 1, act=act)

#     def forward(self, input):
#         """
#         Args:
#             inputs: input images.

#         Returns:
#             Tuple[Tensor]: FPN feature.
#         """

#         #  backbone
#         out_features = self.backbone(input)
#         features = [out_features[f] for f in self.in_features]
#         [x2, x1, x0] = features

#         fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
#         f_out0 = self.upsample(fpn_out0)  # 512/16
#         f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
#         f_out0 = self.C3_p4(f_out0)  # 1024->512/16

#         fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
#         f_out1 = self.upsample(fpn_out1)  # 256/8
#         f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
#         pan_out2 = self.C3_p3(f_out1)  # 512->256/8

#         p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
#         p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
#         pan_out1 = self.C3_n3(p_out1)  # 512->512/16

#         p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
#         p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
#         pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

#         ### after module creation
#         pan_out2_p3 = self.conv1x1_2(pan_out2)
#         pan_out1_n3 = self.conv1x1_1(pan_out1)
#         pan_out0_n4 = self.conv1x1_0(pan_out0)        
        
#         # pan_out2_p3 = torch.mean(pan_out2, dim=1, keepdim = True)
#         # pan_out1_n3 = torch.mean(pan_out1, dim=1, keepdim = True)
#         # pan_out0_n4 = torch.mean(pan_out0, dim=1, keepdim = True)
        
#         pan_out2_p3 = self.sigmoid(pan_out2_p3)
#         pan_out1_n3 = self.sigmoid(pan_out1_n3)
#         pan_out0_n4 = self.sigmoid(pan_out0_n4)
#         #print(pan_out2_p3.shape, pan_out1_n3.shape, pan_out0_n4.shape)
        
#         pan_out2 = pan_out2 * self.upsample(pan_out1_n3)
#         pan_out1 = pan_out1 * self.upsample(pan_out0_n4)
#         pan_out0 = pan_out0 * self.downsample(self.downsample(pan_out2_p3))
    
#         outputs = (pan_out2, pan_out1, pan_out0)
#         return outputs
    

# class YOLOPAFPN(nn.Module):
#     """
#     YOLOv3 model. Darknet 53 is the default backbone of this model.
#     """

#     def __init__(
#         self,
#         depth=1.0,
#         width=1.0,
#         in_features=("dark2", "dark3", "dark4", "dark5"),
#         in_channels=[256, 512, 1024],
#         depthwise=False,
#         act="silu",
#     ):
#         super().__init__()
#         self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
#         self.in_features = in_features
#         self.in_channels = in_channels
#         Conv = DWConv if depthwise else BaseConv

#         self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
#         self.lateral_conv0 = BaseConv(
#             int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
#         )
        
        
#         self.downsample = Conv(
#             1, 1, 3, 2, act=act
#         )
        
#         self.sigmoid = nn.Sigmoid()
        
        
#         self.C3_p4 = CSPLayer(
#             int(2 * in_channels[1] * width)+48,
#             int(in_channels[1] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )  # cat

#         self.reduce_conv1 = BaseConv(
#             int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
#         )
#         self.C3_p3 = CSPLayer(
#             int(2 * in_channels[0] * width)+48,
#             int(in_channels[0] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         # bottom-up conv
#         self.bu_conv2 = Conv(
#             int(in_channels[0] * width), int(in_channels[0] * (width) / 2), 3, 2, act=act
#         )
#         self.bu_dlconv2 = DilateConv(
#             int(in_channels[0] * width), int(in_channels[0] * (width) / 2), 3, 2, act=act
#         )
#         self.C3_n3 = CSPLayer(
#             int(2 * in_channels[0] * width),
#             int(in_channels[1] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         # bottom-up conv
#         self.bu_conv1 = Conv(
#             int(in_channels[1] * width), int(in_channels[1] * (width) / 2), 3, 2, act=act
#         )
#         self.bu_dlconv1 = DilateConv(
#             int(in_channels[1] * width), int(in_channels[1] * (width) / 2), 3, 2, act=act
#         )
#         self.C3_n4 = CSPLayer(
#             int(2 * in_channels[1] * width),
#             int(in_channels[2] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         self.conv1x1_bu2 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
#         self.conv1x1_bu1 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
#         self.conv1x1_bu0 = BaseConv(int(in_channels[2] * width)*2, int(in_channels[2] * width), 1, 1, act=act)

#         # self.conv1x1_2 = BaseConv(int(in_channels[0] * width), 1, 1, 1, act=act)
#         # self.conv1x1_1 = BaseConv(int(in_channels[1] * width), 1, 1, 1, act=act)
#         # self.conv1x1_0 = BaseConv(int(in_channels[2] * width), 1, 1, 1, act=act)
    

#     def forward(self, input):
#         """
#         Args:
#             inputs: input images.

#         Returns:
#             Tuple[Tensor]: FPN feature.
#         """
#         #DICONV_featuremap = Before_Module.DICONV
#         #print(DICONV_featuremap.shape)
#         #  backbone
#         out_features = self.backbone(input)
#         features = [out_features[f] for f in self.in_features]
#         [x3, x2, x1, x0] = features
#         #print(x2.shape, x1.shape, x0.shape)
        
#         fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
#         f_out0 = self.upsample(fpn_out0)  # 512/16
#         f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
#         b,c,w,h = f_out0.shape
#         x3_resize2 = F.interpolate(x3, size=(w,h), mode='bilinear', align_corners=False)
#         f_out0 = torch.cat([f_out0, x3_resize2], 1)
#         f_out0 = self.C3_p4(f_out0)  # 1024->512/16

#         fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
#         f_out1 = self.upsample(fpn_out1)  # 256/8
#         f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
#         b,c,w,h = f_out1.shape
#         x3_resize1 = F.interpolate(x3, size=(w,h), mode='bilinear', align_corners=False)
#         f_out1 = torch.cat([f_out1, x3_resize1], 1)
#         pan_out2 = self.C3_p3(f_out1)  # 512->256/8
#         pan_out2 = torch.cat([pan_out2, x2], 1)
#         pan_out2 = self.conv1x1_bu2(pan_out2)

#         p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
#         p_dlout1 = self.bu_dlconv2(pan_out2)  # 256->256/16
#         p_out1 = torch.cat([p_out1, p_dlout1], 1)
#         p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
#         pan_out1 = self.C3_n3(p_out1)  # 512->512/16
#         pan_out1 = torch.cat([pan_out1, x1], 1)
#         pan_out1 = self.conv1x1_bu1(pan_out1)

#         p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
#         p_dlout0 = self.bu_dlconv1(pan_out1)  # 512->512/32
#         p_out0 = torch.cat([p_out0, p_dlout0], 1)
#         p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
#         pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
#         pan_out0 = torch.cat([pan_out0, x0], 1)
#         pan_out0 = self.conv1x1_bu0(pan_out0)
        
        
#         ### after module creation
#         pan_out2_p3 = self.conv1x1_2(pan_out2)
#         pan_out1_n3 = self.conv1x1_1(pan_out1)
#         pan_out0_n4 = self.conv1x1_0(pan_out0)        
        
#         # pan_out2_p3 = torch.mean(pan_out2, dim=1, keepdim = True)
#         # pan_out1_n3 = torch.mean(pan_out1, dim=1, keepdim = True)
#         # pan_out0_n4 = torch.mean(pan_out0, dim=1, keepdim = True)
        
#         pan_out2_p3 = self.sigmoid(pan_out2_p3)
#         pan_out1_n3 = self.sigmoid(pan_out1_n3)
#         pan_out0_n4 = self.sigmoid(pan_out0_n4)
#         #print(pan_out2_p3.shape, pan_out1_n3.shape, pan_out0_n4.shape)
        
#         pan_out2 = pan_out2 * self.upsample(pan_out1_n3)
#         pan_out1 = pan_out1 * self.upsample(pan_out0_n4)
#         pan_out0 = pan_out0 * self.downsample(self.downsample(pan_out2_p3))
    
#         outputs = (pan_out2, pan_out1, pan_out0)
#         #print(outputs[0].shape, outputs[1].shape, outputs[2].shape)
#         return outputs


# class YOLOPAFPN(nn.Module):
#     """
#     YOLOv3 model. Darknet 53 is the default backbone of this model.
#     """

#     def __init__(
#         self,
#         depth=1.0,
#         width=1.0,
#         in_features=("dark2", "dark3", "dark4", "dark5"),
#         in_channels=[256, 512, 1024],
#         depthwise=False,
#         act="silu",
#     ):
#         super().__init__()
#         self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
#         self.in_features = in_features
#         self.in_channels = in_channels
#         Conv = DWConv if depthwise else BaseConv

#         self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
#         self.lateral_conv0 = BaseConv(
#             int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
#         )


#         self.downsample = Conv(
#             1, 1, 3, 2, act=act
#         )

#         self.sigmoid = nn.Sigmoid()


#         self.C3_p4 = CSPLayer(
#             int(2 * in_channels[1] * width)+48,
#             int(in_channels[1] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )  # cat

#         self.reduce_conv1 = BaseConv(
#             int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
#         )
#         self.C3_p3 = CSPLayer(
#             int(2 * in_channels[0] * width)+48,
#             int(in_channels[0] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         # bottom-up conv
#         # self.bu_conv2 = Conv(
#         #     int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
#         # )
#         self.bu_conv2 = Conv(
#             int(in_channels[0] * width), int(in_channels[0] * (width) / 2), 3, 2, act=act
#         )
#         self.bu_dlconv2 = DilateConv(
#             int(in_channels[0] * width), int(in_channels[0] * (width) / 2), 3, 2, act=act
#         )
#         self.C3_n3 = CSPLayer(
#             int(2 * in_channels[0] * width),
#             int(in_channels[1] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         # bottom-up conv
#         # self.bu_conv1 = Conv(
#         #     int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
#         # )
#         self.bu_conv1 = Conv(
#             int(in_channels[1] * width), int(in_channels[1] * (width) / 2), 3, 2, act=act
#         )
#         self.bu_dlconv1 = DilateConv(
#             int(in_channels[1] * width), int(in_channels[1] * (width) / 2), 3, 2, act=act
#         )
#         self.C3_n4 = CSPLayer(
#             int(2 * in_channels[1] * width),
#             int(in_channels[2] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         self.conv1x1_bu2 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
#         self.conv1x1_bu1 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
#         self.conv1x1_bu0 = BaseConv(int(in_channels[2] * width)*2, int(in_channels[2] * width), 1, 1, act=act)

#         self.conv1x1_2 = BaseConv(int(in_channels[0] * width), 1, 1, 1, act=act)
#         self.conv1x1_1 = BaseConv(int(in_channels[1] * width), 1, 1, 1, act=act)
#         self.conv1x1_0 = BaseConv(int(in_channels[2] * width), 1, 1, 1, act=act)


#     def forward(self, input):
#         """
#         Args:
#             inputs: input images.

#         Returns:
#             Tuple[Tensor]: FPN feature.
#         """
#         #DICONV_featuremap = Before_Module.DICONV
#         #print(DICONV_featuremap.shape)
#         #  backbone
#         out_features = self.backbone(input)
#         features = [out_features[f] for f in self.in_features]
#         [x3, x2, x1, x0] = features
#         #print(x2.shape, x1.shape, x0.shape)

#         fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
#         f_out0 = self.upsample(fpn_out0)  # 512/16
#         f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
#         b,c,w,h = f_out0.shape
#         x3_resize2 = F.interpolate(x3, size=(w,h), mode='bilinear', align_corners=False)
#         f_out0 = torch.cat([f_out0, x3_resize2], 1)
#         f_out0 = self.C3_p4(f_out0)  # 1024->512/16

#         fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
#         f_out1 = self.upsample(fpn_out1)  # 256/8
#         f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
#         b,c,w,h = f_out1.shape
#         x3_resize1 = F.interpolate(x3, size=(w,h), mode='bilinear', align_corners=False)
#         f_out1 = torch.cat([f_out1, x3_resize1], 1)
#         pan_out2 = self.C3_p3(f_out1)  # 512->256/8
#         # pan_out2 = torch.cat([pan_out2, x2], 1)
#         # pan_out2 = self.conv1x1_bu2(pan_out2)

#         p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
#         p_dlout1 = self.bu_dlconv2(pan_out2)  # 256->256/16
#         p_out1 = torch.cat([p_out1, p_dlout1], 1)
#         p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
#         pan_out1 = self.C3_n3(p_out1)  # 512->512/16
#         # pan_out1 = torch.cat([pan_out1, x1], 1)
#         # pan_out1 = self.conv1x1_bu1(pan_out1)

#         p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
#         p_dlout0 = self.bu_dlconv1(pan_out1)  # 512->512/32
#         p_out0 = torch.cat([p_out0, p_dlout0], 1)
#         p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
#         pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
#         # pan_out0 = torch.cat([pan_out0, x0], 1)
#         # pan_out0 = self.conv1x1_bu0(pan_out0)


#         ### after module creation
#         pan_out2_p3 = self.conv1x1_2(pan_out2)
#         pan_out1_n3 = self.conv1x1_1(pan_out1)
#         pan_out0_n4 = self.conv1x1_0(pan_out0)

#         # pan_out2_p3 = torch.mean(pan_out2, dim=1, keepdim = True)
#         # pan_out1_n3 = torch.mean(pan_out1, dim=1, keepdim = True)
#         # pan_out0_n4 = torch.mean(pan_out0, dim=1, keepdim = True)

#         pan_out2_p3 = self.sigmoid(pan_out2_p3)
#         pan_out1_n3 = self.sigmoid(pan_out1_n3)
#         pan_out0_n4 = self.sigmoid(pan_out0_n4)
#         #print(pan_out2_p3.shape, pan_out1_n3.shape, pan_out0_n4.shape)

#         pan_out2 = pan_out2 * self.upsample(pan_out1_n3)
#         pan_out1 = pan_out1 * self.upsample(pan_out0_n4)
#         pan_out0 = pan_out0 * self.downsample(self.downsample(pan_out2_p3))

#         outputs = (pan_out2, pan_out1, pan_out0)
#         #print(outputs[0].shape, outputs[1].shape, outputs[2].shape)
#         return outputs
    


# class YOLOPAFPN(nn.Module):
#     """
#     YOLOv3 model. Darknet 53 is the default backbone of this model.
#     """

#     def __init__(
#         self,
#         depth=1.0,
#         width=1.0,
#         in_features=("dark2", "dark3", "dark4", "dark5"),
#         in_channels=[256, 512, 1024],
#         depthwise=False,
#         act="silu",
#     ):
#         super().__init__()
#         self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
#         self.in_features = in_features
#         self.in_channels = in_channels
#         Conv = DWConv if depthwise else BaseConv

#         self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
#         self.lateral_conv0 = BaseConv(
#             int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
#         )


#         self.downsample = Conv(
#             1, 1, 3, 2, act=act
#         )

#         self.sigmoid = nn.Sigmoid()


#         self.C3_p4 = CSPLayer(
#             int(2 * in_channels[1] * width)+48,
#             int(in_channels[1] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )  # cat

#         self.reduce_conv1 = BaseConv(
#             int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
#         )
#         self.C3_p3 = CSPLayer(
#             int(2 * in_channels[0] * width)+48,
#             int(in_channels[0] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         # bottom-up conv
#         self.bu_conv2 = Conv(
#             int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
#         )
#         # self.bu_conv2 = Conv(
#         #     int(in_channels[0] * width), int(in_channels[0] * (width) / 2), 3, 2, act=act
#         # )
#         # self.bu_dlconv2 = DilateConv(
#         #     int(in_channels[0] * width), int(in_channels[0] * (width) / 2), 3, 2, act=act
#         # )
#         self.C3_n3 = CSPLayer(
#             int(2 * in_channels[0] * width),
#             int(in_channels[1] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         # bottom-up conv
#         self.bu_conv1 = Conv(
#             int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
#         )
#         # self.bu_conv1 = Conv(
#         #     int(in_channels[1] * width), int(in_channels[1] * (width) / 2), 3, 2, act=act
#         # )
#         # self.bu_dlconv1 = DilateConv(
#         #     int(in_channels[1] * width), int(in_channels[1] * (width) / 2), 3, 2, act=act
#         # )
#         self.C3_n4 = CSPLayer(
#             int(2 * in_channels[1] * width),
#             int(in_channels[2] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         self.conv1x1_bu2 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
#         self.conv1x1_bu1 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
#         self.conv1x1_bu0 = BaseConv(int(in_channels[2] * width)*2, int(in_channels[2] * width), 1, 1, act=act)

#         self.conv1x1_2 = BaseConv(int(in_channels[0] * width), 1, 1, 1, act=act)
#         self.conv1x1_1 = BaseConv(int(in_channels[1] * width), 1, 1, 1, act=act)
#         self.conv1x1_0 = BaseConv(int(in_channels[2] * width), 1, 1, 1, act=act)


#     def forward(self, input):
#         """
#         Args:
#             inputs: input images.

#         Returns:
#             Tuple[Tensor]: FPN feature.
#         """
#         #DICONV_featuremap = Before_Module.DICONV
#         #print(DICONV_featuremap.shape)
#         #  backbone
#         out_features = self.backbone(input)
#         features = [out_features[f] for f in self.in_features]
#         [x3, x2, x1, x0] = features
#         #print(x2.shape, x1.shape, x0.shape)

#         fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
#         f_out0 = self.upsample(fpn_out0)  # 512/16
#         f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
#         b,c,w,h = f_out0.shape
#         x3_resize2 = F.interpolate(x3, size=(w,h), mode='bilinear', align_corners=False)
#         f_out0 = torch.cat([f_out0, x3_resize2], 1)
#         f_out0 = self.C3_p4(f_out0)  # 1024->512/16

#         fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
#         f_out1 = self.upsample(fpn_out1)  # 256/8
#         f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
#         b,c,w,h = f_out1.shape
#         x3_resize1 = F.interpolate(x3, size=(w,h), mode='bilinear', align_corners=False)
#         f_out1 = torch.cat([f_out1, x3_resize1], 1)
#         pan_out2 = self.C3_p3(f_out1)  # 512->256/8
#         pan_out2 = torch.cat([pan_out2, x2], 1)
#         pan_out2 = self.conv1x1_bu2(pan_out2)

#         p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
#         # p_dlout1 = self.bu_dlconv2(pan_out2)  # 256->256/16
#         # p_out1 = torch.cat([p_out1, p_dlout1], 1)
#         p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
#         pan_out1 = self.C3_n3(p_out1)  # 512->512/16
#         pan_out1 = torch.cat([pan_out1, x1], 1)
#         pan_out1 = self.conv1x1_bu1(pan_out1)

#         p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
#         # p_dlout0 = self.bu_dlconv1(pan_out1)  # 512->512/32
#         # p_out0 = torch.cat([p_out0, p_dlout0], 1)
#         p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
#         pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
#         pan_out0 = torch.cat([pan_out0, x0], 1)
#         pan_out0 = self.conv1x1_bu0(pan_out0)


#         ### after module creation
#         pan_out2_p3 = self.conv1x1_2(pan_out2)
#         pan_out1_n3 = self.conv1x1_1(pan_out1)
#         pan_out0_n4 = self.conv1x1_0(pan_out0)

#         # pan_out2_p3 = torch.mean(pan_out2, dim=1, keepdim = True)
#         # pan_out1_n3 = torch.mean(pan_out1, dim=1, keepdim = True)
#         # pan_out0_n4 = torch.mean(pan_out0, dim=1, keepdim = True)

#         pan_out2_p3 = self.sigmoid(pan_out2_p3)
#         pan_out1_n3 = self.sigmoid(pan_out1_n3)
#         pan_out0_n4 = self.sigmoid(pan_out0_n4)
#         #print(pan_out2_p3.shape, pan_out1_n3.shape, pan_out0_n4.shape)

#         pan_out2 = pan_out2 * self.upsample(pan_out1_n3)
#         pan_out1 = pan_out1 * self.upsample(pan_out0_n4)
#         pan_out0 = pan_out0 * self.downsample(self.downsample(pan_out2_p3))

#         outputs = (pan_out2, pan_out1, pan_out0)
#         #print(outputs[0].shape, outputs[1].shape, outputs[2].shape)
#         return outputs


# class YOLOPAFPN(nn.Module):
#     """
#     YOLOv3 model. Darknet 53 is the default backbone of this model.
#     """

#     def __init__(
#         self,
#         depth=1.0,
#         width=1.0,
#         in_features=("dark2", "dark3", "dark4", "dark5"),
#         in_channels=[256, 512, 1024],
#         depthwise=False,
#         act="silu",
#     ):
#         super().__init__()
#         self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
#         self.in_features = in_features
#         self.in_channels = in_channels
#         Conv = DWConv if depthwise else BaseConv

#         self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
#         self.lateral_conv0 = BaseConv(
#             int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
#         )


#         self.downsample = Conv(
#             1, 1, 3, 2, act=act
#         )

#         self.sigmoid = nn.Sigmoid()


#         self.C3_p4 = CSPLayer(
#             int(2 * in_channels[1] * width)+48,
#             int(in_channels[1] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )  # cat

#         self.reduce_conv1 = BaseConv(
#             int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
#         )
#         self.C3_p3 = CSPLayer(
#             int(2 * in_channels[0] * width)+48,
#             int(in_channels[0] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         # bottom-up conv
#         self.bu_conv2 = Conv(
#             int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
#         )
#         # self.bu_conv2 = Conv(
#         #     int(in_channels[0] * width), int(in_channels[0] * (width) / 2), 3, 2, act=act
#         # )
#         # self.bu_dlconv2 = DilateConv(
#         #     int(in_channels[0] * width), int(in_channels[0] * (width) / 2), 3, 2, act=act
#         # )
#         self.C3_n3 = CSPLayer(
#             int(2 * in_channels[0] * width),
#             int(in_channels[1] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         # bottom-up conv
#         self.bu_conv1 = Conv(
#             int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
#         )
#         # self.bu_conv1 = Conv(
#         #     int(in_channels[1] * width), int(in_channels[1] * (width) / 2), 3, 2, act=act
#         # )
#         # self.bu_dlconv1 = DilateConv(
#         #     int(in_channels[1] * width), int(in_channels[1] * (width) / 2), 3, 2, act=act
#         # )
#         self.C3_n4 = CSPLayer(
#             int(2 * in_channels[1] * width),
#             int(in_channels[2] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         self.conv1x1_bu2 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
#         self.conv1x1_bu1 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
#         self.conv1x1_bu0 = BaseConv(int(in_channels[2] * width)*2, int(in_channels[2] * width), 1, 1, act=act)

#         self.conv1x1_2 = BaseConv(int(in_channels[0] * width), 1, 1, 1, act=act)
#         self.conv1x1_1 = BaseConv(int(in_channels[1] * width), 1, 1, 1, act=act)
#         self.conv1x1_0 = BaseConv(int(in_channels[2] * width), 1, 1, 1, act=act)


#     def forward(self, input):
#         """
#         Args:
#             inputs: input images.

#         Returns:
#             Tuple[Tensor]: FPN feature.
#         """
#         #DICONV_featuremap = Before_Module.DICONV
#         #print(DICONV_featuremap.shape)
#         #  backbone
#         out_features = self.backbone(input)
#         features = [out_features[f] for f in self.in_features]
#         [x3, x2, x1, x0] = features
#         #print(x2.shape, x1.shape, x0.shape)

#         fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
#         f_out0 = self.upsample(fpn_out0)  # 512/16
#         f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
#         b,c,w,h = f_out0.shape
#         x3_resize2 = F.interpolate(x3, size=(w,h), mode='bilinear', align_corners=False)
#         f_out0 = torch.cat([f_out0, x3_resize2], 1)
#         f_out0 = self.C3_p4(f_out0)  # 1024->512/16

#         fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
#         f_out1 = self.upsample(fpn_out1)  # 256/8
#         f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
#         b,c,w,h = f_out1.shape
#         x3_resize1 = F.interpolate(x3, size=(w,h), mode='bilinear', align_corners=False)
#         f_out1 = torch.cat([f_out1, x3_resize1], 1)
#         pan_out2 = self.C3_p3(f_out1)  # 512->256/8
#         # pan_out2 = torch.cat([pan_out2, x2], 1)
#         # pan_out2 = self.conv1x1_bu2(pan_out2)

#         p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
#         # p_dlout1 = self.bu_dlconv2(pan_out2)  # 256->256/16
#         # p_out1 = torch.cat([p_out1, p_dlout1], 1)
#         p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
#         pan_out1 = self.C3_n3(p_out1)  # 512->512/16
#         # pan_out1 = torch.cat([pan_out1, x1], 1)
#         # pan_out1 = self.conv1x1_bu1(pan_out1)

#         p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
#         # p_dlout0 = self.bu_dlconv1(pan_out1)  # 512->512/32
#         # p_out0 = torch.cat([p_out0, p_dlout0], 1)
#         p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
#         pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
#         # pan_out0 = torch.cat([pan_out0, x0], 1)
#         # pan_out0 = self.conv1x1_bu0(pan_out0)


#         ### after module creation
#         pan_out2_p3 = self.conv1x1_2(pan_out2)
#         pan_out1_n3 = self.conv1x1_1(pan_out1)
#         pan_out0_n4 = self.conv1x1_0(pan_out0)

#         # pan_out2_p3 = torch.mean(pan_out2, dim=1, keepdim = True)
#         # pan_out1_n3 = torch.mean(pan_out1, dim=1, keepdim = True)
#         # pan_out0_n4 = torch.mean(pan_out0, dim=1, keepdim = True)

#         pan_out2_p3 = self.sigmoid(pan_out2_p3)
#         pan_out1_n3 = self.sigmoid(pan_out1_n3)
#         pan_out0_n4 = self.sigmoid(pan_out0_n4)
#         #print(pan_out2_p3.shape, pan_out1_n3.shape, pan_out0_n4.shape)

#         pan_out2 = pan_out2 * self.upsample(pan_out1_n3)
#         pan_out1 = pan_out1 * self.upsample(pan_out0_n4)
#         pan_out0 = pan_out0 * self.downsample(self.downsample(pan_out2_p3))

#         outputs = (pan_out2, pan_out1, pan_out0)
#         #print(outputs[0].shape, outputs[1].shape, outputs[2].shape)
#         return outputs


# class YOLOPAFPN(nn.Module):
#     """
#     YOLOv3 model. Darknet 53 is the default backbone of this model.
#     """

#     def __init__(
#         self,
#         depth=1.0,
#         width=1.0,
#         in_features=("dark2", "dark3", "dark4", "dark5"),
#         in_channels=[256, 512, 1024],
#         depthwise=False,
#         act="silu",
#     ):
#         super().__init__()
#         self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
#         self.in_features = in_features
#         self.in_channels = in_channels
#         Conv = DWConv if depthwise else BaseConv

#         self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
#         self.lateral_conv0 = BaseConv(
#             int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
#         )


#         # self.downsample = Conv(
#         #     1, 1, 3, 2, act=act
#         # )

#         # self.sigmoid = nn.Sigmoid()


#         self.C3_p4 = CSPLayer(
#             int(2 * in_channels[1] * width),
#             int(in_channels[1] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )  # cat

#         self.reduce_conv1 = BaseConv(
#             int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
#         )
#         self.C3_p3 = CSPLayer(
#             int(2 * in_channels[0] * width),
#             int(in_channels[0] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         # bottom-up conv
#         self.bu_conv2 = Conv(
#             int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
#         )
#         # self.bu_conv2 = Conv(
#         #     int(in_channels[0] * width), int(in_channels[0] * (width) / 2), 3, 2, act=act
#         # )
#         # self.bu_dlconv2 = DilateConv(
#         #     int(in_channels[0] * width), int(in_channels[0] * (width) / 2), 3, 2, act=act
#         # )
#         self.C3_n3 = CSPLayer(
#             int(2 * in_channels[0] * width),
#             int(in_channels[1] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         # bottom-up conv
#         self.bu_conv1 = Conv(
#             int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
#         )
#         # self.bu_conv1 = Conv(
#         #     int(in_channels[1] * width), int(in_channels[1] * (width) / 2), 3, 2, act=act
#         # )
#         # self.bu_dlconv1 = DilateConv(
#         #     int(in_channels[1] * width), int(in_channels[1] * (width) / 2), 3, 2, act=act
#         # )
#         self.C3_n4 = CSPLayer(
#             int(2 * in_channels[1] * width),
#             int(in_channels[2] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         # self.conv1x1_bu2 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
#         # self.conv1x1_bu1 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
#         # self.conv1x1_bu0 = BaseConv(int(in_channels[2] * width)*2, int(in_channels[2] * width), 1, 1, act=act)

#         # self.conv1x1_2 = BaseConv(int(in_channels[0] * width), 1, 1, 1, act=act)
#         # self.conv1x1_1 = BaseConv(int(in_channels[1] * width), 1, 1, 1, act=act)
#         # self.conv1x1_0 = BaseConv(int(in_channels[2] * width), 1, 1, 1, act=act)


#     def forward(self, input):
#         """
#         Args:
#             inputs: input images.

#         Returns:
#             Tuple[Tensor]: FPN feature.
#         """
#         #DICONV_featuremap = Before_Module.DICONV
#         #print(DICONV_featuremap.shape)
#         #  backbone
#         out_features = self.backbone(input)
#         features = [out_features[f] for f in self.in_features]
#         [x3, x2, x1, x0] = features
#         #print(x2.shape, x1.shape, x0.shape)

#         fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
#         f_out0 = self.upsample(fpn_out0)  # 512/16
#         f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
#         f_out0 = self.C3_p4(f_out0)  # 1024->512/16

#         fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
#         f_out1 = self.upsample(fpn_out1)  # 256/8

#         f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
#         pan_out2 = self.C3_p3(f_out1)  # 512->256/8

#         p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
#         p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
#         pan_out1 = self.C3_n3(p_out1)  # 512->512/16


#         p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
#         p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
#         pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32


#         # ### after module creation
#         # pan_out2_p3 = self.conv1x1_2(pan_out2)
#         # pan_out1_n3 = self.conv1x1_1(pan_out1)
#         # pan_out0_n4 = self.conv1x1_0(pan_out0)

#         # # pan_out2_p3 = torch.mean(pan_out2, dim=1, keepdim = True)
#         # # pan_out1_n3 = torch.mean(pan_out1, dim=1, keepdim = True)
#         # # pan_out0_n4 = torch.mean(pan_out0, dim=1, keepdim = True)

#         # pan_out2_p3 = self.sigmoid(pan_out2_p3)
#         # pan_out1_n3 = self.sigmoid(pan_out1_n3)
#         # pan_out0_n4 = self.sigmoid(pan_out0_n4)
#         # print(f'{pan_out2_p3.shape} {pan_out1_n3.shape} {pan_out0_n4.shape}')
#         # #print(pan_out2_p3.shape, pan_out1_n3.shape, pan_out0_n4.shape)

#         # pan_out2 = pan_out2 * self.upsample(pan_out1_n3)
#         # pan_out1 = pan_out1 * self.upsample(pan_out0_n4)
#         # pan_out0 = pan_out0 * self.downsample(self.downsample(pan_out2_p3))

#         outputs = (pan_out2, pan_out1, pan_out0)
#         #print(outputs[0].shape, outputs[1].shape, outputs[2].shape)
#         return outputs


# class YOLOPAFPN(nn.Module):
#     """
#     YOLOv3 model. Darknet 53 is the default backbone of this model.
#     """

#     def __init__(
#         self,
#         depth=1.0,
#         width=1.0,
#         in_features=("dark2", "dark3", "dark4", "dark5"),
#         in_channels=[256, 512, 1024],
#         depthwise=False,
#         act="silu",
#     ):
#         super().__init__()
#         self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
#         self.in_features = in_features
#         self.in_channels = in_channels
#         Conv = DWConv if depthwise else BaseConv

#         self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
#         self.lateral_conv0 = BaseConv(
#             int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
#         )


#         self.downsample = Conv(
#             1, 1, 3, 2, act=act
#         )

#         self.sigmoid = nn.Sigmoid()


#         self.C3_p4 = CSPLayer(
#             int(2 * in_channels[1] * width)+48,
#             int(in_channels[1] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )  # cat

#         self.reduce_conv1 = BaseConv(
#             int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
#         )
#         self.C3_p3 = CSPLayer(
#             int(2 * in_channels[0] * width)+48,
#             int(in_channels[0] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         # bottom-up conv
#         self.bu_conv2 = Conv(
#             int(in_channels[0] * width), int(in_channels[0] * (width) / 2), 3, 2, act=act
#         )
#         self.bu_dlconv2 = DilateConv(
#             int(in_channels[0] * width), int(in_channels[0] * (width) / 2), 3, 2, act=act
#         )
#         self.C3_n3 = CSPLayer(
#             int(2 * in_channels[0] * width),
#             int(in_channels[1] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         # bottom-up conv
#         self.bu_conv1 = Conv(
#             int(in_channels[1] * width), int(in_channels[1] * (width) / 2), 3, 2, act=act
#         )
#         self.bu_dlconv1 = DilateConv(
#             int(in_channels[1] * width), int(in_channels[1] * (width) / 2), 3, 2, act=act
#         )
#         self.C3_n4 = CSPLayer(
#             int(2 * in_channels[1] * width),
#             int(in_channels[2] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         self.conv1x1_bu2 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
#         self.conv1x1_bu1 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
#         self.conv1x1_bu0 = BaseConv(int(in_channels[2] * width)*2, int(in_channels[2] * width), 1, 1, act=act)

#         self.conv1x1_2 = BaseConv(int(in_channels[0] * width), 1, 1, 1, act=act)
#         self.conv1x1_1 = BaseConv(int(in_channels[1] * width), 1, 1, 1, act=act)
#         self.conv1x1_0 = BaseConv(int(in_channels[2] * width), 1, 1, 1, act=act)


#     def forward(self, input):
#         """
#         Args:
#             inputs: input images.

#         Returns:
#             Tuple[Tensor]: FPN feature.
#         """
#         #DICONV_featuremap = Before_Module.DICONV
#         #print(DICONV_featuremap.shape)
#         #  backbone
#         out_features = self.backbone(input)
#         features = [out_features[f] for f in self.in_features]
#         [x3, x2, x1, x0] = features
#         #print(x2.shape, x1.shape, x0.shape)

#         fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
#         f_out0 = self.upsample(fpn_out0)  # 512/16
#         f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
#         b,c,w,h = f_out0.shape
#         x3_resize2 = F.interpolate(x3, size=(w,h), mode='bilinear', align_corners=False)
#         f_out0 = torch.cat([f_out0, x3_resize2], 1)
#         f_out0 = self.C3_p4(f_out0)  # 1024->512/16

#         fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
#         f_out1 = self.upsample(fpn_out1)  # 256/8
#         f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
#         b,c,w,h = f_out1.shape
#         x3_resize1 = F.interpolate(x3, size=(w,h), mode='bilinear', align_corners=False)
#         f_out1 = torch.cat([f_out1, x3_resize1], 1)
#         pan_out2 = self.C3_p3(f_out1)  # 512->256/8
#         pan_out2 = torch.cat([pan_out2, x2], 1)
#         pan_out2 = self.conv1x1_bu2(pan_out2)

#         p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
#         p_dlout1 = self.bu_dlconv2(pan_out2)  # 256->256/16
#         p_out1 = torch.cat([p_out1, p_dlout1], 1)
#         p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
#         pan_out1 = self.C3_n3(p_out1)  # 512->512/16
#         pan_out1 = torch.cat([pan_out1, x1], 1)
#         pan_out1 = self.conv1x1_bu1(pan_out1)

#         p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
#         p_dlout0 = self.bu_dlconv1(pan_out1)  # 512->512/32
#         p_out0 = torch.cat([p_out0, p_dlout0], 1)
#         p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
#         pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
#         pan_out0 = torch.cat([pan_out0, x0], 1)
#         pan_out0 = self.conv1x1_bu0(pan_out0)


#         # ### after module creation
#         # pan_out2_p3 = self.conv1x1_2(pan_out2)
#         # pan_out1_n3 = self.conv1x1_1(pan_out1)
#         # pan_out0_n4 = self.conv1x1_0(pan_out0)

#         # # pan_out2_p3 = torch.mean(pan_out2, dim=1, keepdim = True)
#         # # pan_out1_n3 = torch.mean(pan_out1, dim=1, keepdim = True)
#         # # pan_out0_n4 = torch.mean(pan_out0, dim=1, keepdim = True)

#         # pan_out2_p3 = self.sigmoid(pan_out2_p3)
#         # pan_out1_n3 = self.sigmoid(pan_out1_n3)
#         # pan_out0_n4 = self.sigmoid(pan_out0_n4)
#         # print(f'{pan_out2_p3.shape} {pan_out1_n3.shape} {pan_out0_n4.shape}')
#         # #print(pan_out2_p3.shape, pan_out1_n3.shape, pan_out0_n4.shape)

#         # pan_out2 = pan_out2 * self.upsample(pan_out1_n3)
#         # pan_out1 = pan_out1 * self.upsample(pan_out0_n4)
#         # pan_out0 = pan_out0 * self.downsample(self.downsample(pan_out2_p3))

#         outputs = (pan_out2, pan_out1, pan_out0)
#         #print(outputs[0].shape, outputs[1].shape, outputs[2].shape)
#         return outputs


# class YOLOPAFPN(nn.Module):
#     """
#     YOLOv3 model. Darknet 53 is the default backbone of this model.
#     """

#     def __init__(
#         self,
#         depth=1.0,
#         width=1.0,
#         in_features=("dark2", "dark3", "dark4", "dark5"),
#         in_channels=[256, 512, 1024],
#         depthwise=False,
#         act="silu",
#     ):
#         super().__init__()
#         self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
#         self.in_features = in_features
#         self.in_channels = in_channels
#         Conv = DWConv if depthwise else BaseConv

#         self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
#         self.lateral_conv0 = BaseConv(
#             int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
#         )


#         self.downsample = Conv(
#             1, 1, 3, 2, act=act
#         )

#         self.sigmoid = nn.Sigmoid()


#         self.C3_p4 = CSPLayer(
#             int(2 * in_channels[1] * width)+64,
#             int(in_channels[1] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )  # cat

#         self.reduce_conv1 = BaseConv(
#             int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
#         )
#         self.C3_p3 = CSPLayer(
#             int(2 * in_channels[0] * width)+64,
#             int(in_channels[0] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         # bottom-up conv
#         self.bu_conv2 = Conv(
#             int(in_channels[0] * width), int(in_channels[0] * (width) / 2), 3, 2, act=act
#         )
#         self.bu_dlconv2 = DilateConv(
#             int(in_channels[0] * width), int(in_channels[0] * (width) / 2), 3, 2, act=act
#         )
#         self.C3_n3 = CSPLayer(
#             int(2 * in_channels[0] * width),
#             int(in_channels[1] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         # bottom-up conv
#         self.bu_conv1 = Conv(
#             int(in_channels[1] * width), int(in_channels[1] * (width) / 2), 3, 2, act=act
#         )
#         self.bu_dlconv1 = DilateConv(
#             int(in_channels[1] * width), int(in_channels[1] * (width) / 2), 3, 2, act=act
#         )
#         self.C3_n4 = CSPLayer(
#             int(2 * in_channels[1] * width),
#             int(in_channels[2] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         self.conv1x1_bu2 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
#         self.conv1x1_bu1 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
#         self.conv1x1_bu0 = BaseConv(int(in_channels[2] * width)*2, int(in_channels[2] * width), 1, 1, act=act)

#         self.conv1x1_2 = BaseConv(int(in_channels[0] * width), 1, 1, 1, act=act)
#         self.conv1x1_1 = BaseConv(int(in_channels[1] * width), 1, 1, 1, act=act)
#         self.conv1x1_0 = BaseConv(int(in_channels[2] * width), 1, 1, 1, act=act)


#     def forward(self, input):
#         """
#         Args:
#             inputs: input images.

#         Returns:
#             Tuple[Tensor]: FPN feature.
#         """
#         #DICONV_featuremap = Before_Module.DICONV
#         #print(DICONV_featuremap.shape)
#         #  backbone
#         out_features = self.backbone(input)
#         features = [out_features[f] for f in self.in_features]
#         [x3, x2, x1, x0] = features
#         #print(x2.shape, x1.shape, x0.shape)

#         fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
#         f_out0 = self.upsample(fpn_out0)  # 512/16
#         f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
#         b,c,w,h = f_out0.shape
#         x3_resize2 = F.interpolate(x3, size=(w,h), mode='bilinear', align_corners=False)
#         f_out0 = torch.cat([f_out0, x3_resize2], 1)
#         f_out0 = self.C3_p4(f_out0)  # 1024->512/16

#         fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
#         f_out1 = self.upsample(fpn_out1)  # 256/8
#         f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
#         b,c,w,h = f_out1.shape
#         x3_resize1 = F.interpolate(x3, size=(w,h), mode='bilinear', align_corners=False)
#         f_out1 = torch.cat([f_out1, x3_resize1], 1)
#         pan_out2 = self.C3_p3(f_out1)  # 512->256/8
#         pan_out2 = torch.cat([pan_out2, x2], 1)
#         pan_out2 = self.conv1x1_bu2(pan_out2)

#         p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
#         p_dlout1 = self.bu_dlconv2(pan_out2)  # 256->256/16
#         p_out1 = torch.cat([p_out1, p_dlout1], 1)
#         p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
#         pan_out1 = self.C3_n3(p_out1)  # 512->512/16
#         pan_out1 = torch.cat([pan_out1, x1], 1)
#         pan_out1 = self.conv1x1_bu1(pan_out1)

#         p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
#         p_dlout0 = self.bu_dlconv1(pan_out1)  # 512->512/32
#         p_out0 = torch.cat([p_out0, p_dlout0], 1)
#         p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
#         pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
#         pan_out0 = torch.cat([pan_out0, x0], 1)
#         pan_out0 = self.conv1x1_bu0(pan_out0)


#         ### after module creation
#         pan_out2_p3 = self.conv1x1_2(pan_out2)
#         pan_out1_n3 = self.conv1x1_1(pan_out1)
#         pan_out0_n4 = self.conv1x1_0(pan_out0)

#         # pan_out2_p3 = torch.mean(pan_out2, dim=1, keepdim = True)
#         # pan_out1_n3 = torch.mean(pan_out1, dim=1, keepdim = True)
#         # pan_out0_n4 = torch.mean(pan_out0, dim=1, keepdim = True)

#         pan_out2_p3 = self.sigmoid(pan_out2_p3)
#         pan_out1_n3 = self.sigmoid(pan_out1_n3)
#         pan_out0_n4 = self.sigmoid(pan_out0_n4)
#         # print(f'{pan_out2_p3.shape} {pan_out1_n3.shape} {pan_out0_n4.shape}')
#         #print(pan_out2_p3.shape, pan_out1_n3.shape, pan_out0_n4.shape)

#         pan_out2 = pan_out2 * self.upsample(pan_out1_n3)
#         pan_out1 = pan_out1 * self.upsample(pan_out0_n4)
#         pan_out0 = pan_out0 * self.downsample(self.downsample(pan_out2_p3))

#         outputs = (pan_out2, pan_out1, pan_out0)
#         #print(outputs[0].shape, outputs[1].shape, outputs[2].shape)
#         return outputs

# class YOLOPAFPN(nn.Module):
#     """
#     YOLOv3 model. Darknet 53 is the default backbone of this model.
#     """

#     def __init__(
#         self,
#         depth=1.0,
#         width=1.0,
#         in_features=("dark2", "dark3", "dark4", "dark5"),
#         in_channels=[256, 512, 1024],
#         depthwise=False,
#         act="silu",
#     ):
#         super().__init__()
#         self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
#         self.in_features = in_features
#         self.in_channels = in_channels
#         Conv = DWConv if depthwise else BaseConv

#         self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
#         self.lateral_conv0 = BaseConv(
#             int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
#         )


#         self.downsample = Conv(
#             1, 1, 3, 2, act=act
#         )

#         self.sigmoid = nn.Sigmoid()


#         self.C3_p4 = CSPLayer(
#             int(2 * in_channels[1] * width)+48,
#             int(in_channels[1] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )  # cat

#         self.reduce_conv1 = BaseConv(
#             int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
#         )
#         self.C3_p3 = CSPLayer(
#             int(2 * in_channels[0] * width)+48,
#             int(in_channels[0] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         # bottom-up conv
#         self.bu_conv2 = Conv(
#             int(in_channels[0] * width), int(in_channels[0] * (width) / 2), 3, 2, act=act
#         )
#         self.bu_dlconv2 = DilateConv(
#             int(in_channels[0] * width), int(in_channels[0] * (width) / 2), 3, 2, act=act
#         )
#         self.C3_n3 = CSPLayer(
#             int(2 * in_channels[0] * width),
#             int(in_channels[1] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         # bottom-up conv
#         self.bu_conv1 = Conv(
#             int(in_channels[1] * width), int(in_channels[1] * (width) / 2), 3, 2, act=act
#         )
#         self.bu_dlconv1 = DilateConv(
#             int(in_channels[1] * width), int(in_channels[1] * (width) / 2), 3, 2, act=act
#         )
#         self.C3_n4 = CSPLayer(
#             int(2 * in_channels[1] * width),
#             int(in_channels[2] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         self.conv1x1_bu2 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
#         self.conv1x1_bu1 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
#         self.conv1x1_bu0 = BaseConv(int(in_channels[2] * width)*2, int(in_channels[2] * width), 1, 1, act=act)

#         self.conv1x1_2 = BaseConv(int(in_channels[0] * width), 1, 1, 1, act=act)
#         self.conv1x1_1 = BaseConv(int(in_channels[1] * width), 1, 1, 1, act=act)
#         self.conv1x1_0 = BaseConv(int(in_channels[2] * width), 1, 1, 1, act=act)


#     def forward(self, input):
#         """
#         Args:
#             inputs: input images.

#         Returns:
#             Tuple[Tensor]: FPN feature.
#         """
#         #DICONV_featuremap = Before_Module.DICONV
#         #print(DICONV_featuremap.shape)
#         #  backbone
#         out_features = self.backbone(input)
#         features = [out_features[f] for f in self.in_features]
#         [x3, x2, x1, x0] = features
#         #print(x2.shape, x1.shape, x0.shape)

#         fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
#         f_out0 = self.upsample(fpn_out0)  # 512/16
#         f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
#         b,c,w,h = f_out0.shape
#         x3_resize2 = F.interpolate(x3, size=(w,h), mode='bilinear', align_corners=False)
#         f_out0 = torch.cat([f_out0, x3_resize2], 1)
#         f_out0 = self.C3_p4(f_out0)  # 1024->512/16

#         fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
#         f_out1 = self.upsample(fpn_out1)  # 256/8
#         f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
#         b,c,w,h = f_out1.shape
#         x3_resize1 = F.interpolate(x3, size=(w,h), mode='bilinear', align_corners=False)
#         f_out1 = torch.cat([f_out1, x3_resize1], 1)
#         pan_out2 = self.C3_p3(f_out1)  # 512->256/8
#         pan_out2 = torch.cat([pan_out2, x2], 1)
#         pan_out2 = self.conv1x1_bu2(pan_out2)

#         p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
#         p_dlout1 = self.bu_dlconv2(pan_out2)  # 256->256/16
#         p_out1 = torch.cat([p_out1, p_dlout1], 1)
#         p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
#         pan_out1 = self.C3_n3(p_out1)  # 512->512/16
#         pan_out1 = torch.cat([pan_out1, x1], 1)
#         pan_out1 = self.conv1x1_bu1(pan_out1)

#         p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
#         p_dlout0 = self.bu_dlconv1(pan_out1)  # 512->512/32
#         p_out0 = torch.cat([p_out0, p_dlout0], 1)
#         p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
#         pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
#         pan_out0 = torch.cat([pan_out0, x0], 1)
#         pan_out0 = self.conv1x1_bu0(pan_out0)


#         ### after module creation
#         pan_out2_p3 = self.conv1x1_2(pan_out2)
#         pan_out1_n3 = self.conv1x1_1(pan_out1)
#         pan_out0_n4 = self.conv1x1_0(pan_out0)

#         # pan_out2_p3 = torch.mean(pan_out2, dim=1, keepdim = True)
#         # pan_out1_n3 = torch.mean(pan_out1, dim=1, keepdim = True)
#         # pan_out0_n4 = torch.mean(pan_out0, dim=1, keepdim = True)

#         pan_out2_p3 = self.sigmoid(pan_out2_p3)
#         pan_out1_n3 = self.sigmoid(pan_out1_n3)
#         pan_out0_n4 = self.sigmoid(pan_out0_n4)
#         #print(pan_out2_p3.shape, pan_out1_n3.shape, pan_out0_n4.shape)

#         pan_out2 = pan_out2 * self.upsample(pan_out1_n3)
#         pan_out1 = pan_out1 * self.upsample(pan_out0_n4)
#         pan_out0 = pan_out0 * self.downsample(self.downsample(pan_out2_p3))

#         outputs = (pan_out2, pan_out1, pan_out0)
#         #print(outputs[0].shape, outputs[1].shape, outputs[2].shape)
#         return outputs


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark2", "dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )


        self.downsample = Conv(
            1, 1, 3, 2, act=act
        )

        self.sigmoid = nn.Sigmoid()


        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width)+64,
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width)+64,
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        # self.bu_conv2 = Conv(
        #     int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        # )
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * (width) / 2), 3, 2, act=act
        )
        self.bu_dlconv2 = DilateConv(
            int(in_channels[0] * width), int(in_channels[0] * (width) / 2), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        # self.bu_conv1 = Conv(
        #     int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        # )
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * (width) / 2), 3, 2, act=act
        )
        self.bu_dlconv1 = DilateConv(
            int(in_channels[1] * width), int(in_channels[1] * (width) / 2), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.conv1x1_bu2 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        self.conv1x1_bu1 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
        self.conv1x1_bu0 = BaseConv(int(in_channels[2] * width)*2, int(in_channels[2] * width), 1, 1, act=act)

        self.conv1x1_2 = BaseConv(int(in_channels[0] * width), 1, 1, 1, act=act)
        self.conv1x1_1 = BaseConv(int(in_channels[1] * width), 1, 1, 1, act=act)
        self.conv1x1_0 = BaseConv(int(in_channels[2] * width), 1, 1, 1, act=act)


    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        #DICONV_featuremap = Before_Module.DICONV
        #print(DICONV_featuremap.shape)
        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x3, x2, x1, x0] = features
        #print(x2.shape, x1.shape, x0.shape)

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        b,c,w,h = f_out0.shape
        x3_resize2 = F.interpolate(x3, size=(w,h), mode='bilinear', align_corners=False)
        f_out0 = torch.cat([f_out0, x3_resize2], 1)
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        b,c,w,h = f_out1.shape
        x3_resize1 = F.interpolate(x3, size=(w,h), mode='bilinear', align_corners=False)
        f_out1 = torch.cat([f_out1, x3_resize1], 1)
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8
        pan_out2 = torch.cat([pan_out2, x2], 1)
        pan_out2 = self.conv1x1_bu2(pan_out2)

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_dlout1 = self.bu_dlconv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, p_dlout1], 1)
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16
        pan_out1 = torch.cat([pan_out1, x1], 1)
        pan_out1 = self.conv1x1_bu1(pan_out1)

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_dlout0 = self.bu_dlconv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, p_dlout0], 1)
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
        pan_out0 = torch.cat([pan_out0, x0], 1)
        pan_out0 = self.conv1x1_bu0(pan_out0)


        ### after module creation
        pan_out2_p3 = self.conv1x1_2(pan_out2)
        pan_out1_n3 = self.conv1x1_1(pan_out1)
        pan_out0_n4 = self.conv1x1_0(pan_out0)

        # pan_out2_p3 = torch.mean(pan_out2, dim=1, keepdim = True)
        # pan_out1_n3 = torch.mean(pan_out1, dim=1, keepdim = True)
        # pan_out0_n4 = torch.mean(pan_out0, dim=1, keepdim = True)

        pan_out2_p3 = self.sigmoid(pan_out2_p3)
        pan_out1_n3 = self.sigmoid(pan_out1_n3)
        pan_out0_n4 = self.sigmoid(pan_out0_n4)
        #print(pan_out2_p3.shape, pan_out1_n3.shape, pan_out0_n4.shape)

        pan_out2 = pan_out2 * self.upsample(pan_out1_n3)
        pan_out1 = pan_out1 * self.upsample(pan_out0_n4)
        pan_out0 = pan_out0 * self.downsample(self.downsample(pan_out2_p3))

        outputs = (pan_out2, pan_out1, pan_out0)
        #print(outputs[0].shape, outputs[1].shape, outputs[2].shape)
        return outputs