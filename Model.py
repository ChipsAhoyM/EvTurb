import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.ops import deform_conv2d
import numpy as np


import logging

from Blocks import ResBlock, SAM, LayerNorm2d, EventImageFusion, BaselineBlock


class Event_Guided_Deformable_Conv(nn.Module):
    def __init__(self, channels, kernel_size = 3, groups=1):
        super(Event_Guided_Deformable_Conv, self).__init__()
        dim = kernel_size * kernel_size * groups * 2
        self.proj_conv_structure = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(dim, use_dropout=False)
        )
        self.weight = nn.Parameter(torch.randn(channels, channels // groups, 3, 3))
        self.offset_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.mask_conv = nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x, event_feature):
        event_feature = self.proj_conv_structure(event_feature)
        offset = self.offset_conv(event_feature)
        mask = self.mask_conv(event_feature)
        x = deform_conv2d(x, offset = offset, mask = mask, padding = 1, stride = 1, dilation = 1, weight = self.weight)
        return x
    

class EventEncoder(nn.Module):
    def __init__(self, input_channel = 3, event_channel = 128, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., num_blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        c = input_channel
        # self.conv_init = nn.Conv2d(in_channels= event_channel, out_channels=c, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        for num in range(num_blocks):
           
            if num == 0:
                self.downs.append(nn.Conv2d(in_channels= event_channel, out_channels=c, kernel_size=3, padding=1, stride=1, groups=1, bias=True))
            else:
                self.downs.append(nn.Conv2d(in_channels=c, out_channels=c * DW_Expand, kernel_size=3, padding=1, stride=2, groups=1, bias=True))
                c = c * DW_Expand
            self.blocks.append(BaselineBlock(c, DW_Expand, FFN_Expand, drop_out_rate))
            # print(f'c: {c}')

    def forward(self, x):
        ev_features = []
        for block, down in zip(self.blocks, self.downs):
            x = down(x)
            x = block(x)
            ev_features.append(x)
            # print(f'x shape: {x.shape}')
        return ev_features

class ImgEncoder(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., num_blocks=3, num_heads = [1,2,4]):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ImageFusion = nn.ModuleList()
        self.num_blocks = num_blocks
        

        for num in range(num_blocks):
           
            if num == 0:
                self.downs.append(nn.Conv2d(in_channels=3, out_channels=c, kernel_size=3, padding=1, stride=1, groups=1, bias=True))
            else:
                self.downs.append(nn.Conv2d(in_channels=c, out_channels=c * DW_Expand, kernel_size=3, padding=1, stride=2, groups=1, bias=True))
                c = c * DW_Expand
            self.ImageFusion.append(EventImageFusion(c, num_heads[num]))
            self.blocks.append(BaselineBlock(c, DW_Expand, FFN_Expand, drop_out_rate))
            


    def forward(self, x, ev_features):
        im_features = []
        for block, down, event, fusion in zip(self.blocks, self.downs, ev_features, self.ImageFusion):
            x = down(x)
            x = block(x)
            x = fusion(x, event)
            im_features.append(x)
            # print(f'x shape: {x.shape}')
        return x, im_features
    


class FirstDecoder(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.skip_conv = nn.ModuleList()
        self.num_blocks = num_blocks

        for num in range(num_blocks):
            self.ups.append(nn.ConvTranspose2d(in_channels=c, out_channels=c // DW_Expand, kernel_size=2, stride=2, bias=True))
            c = c // DW_Expand
            self.skip_conv.append(nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=1, bias=True))
            self.blocks.append(BaselineBlock(c, DW_Expand, FFN_Expand, drop_out_rate))
            


    def forward(self, x, im_features):
        # print(f'x shape: {x.shape}')
        # output_features = []
        for idx in range(self.num_blocks):
            x = self.ups[idx](x)
            x = x +self.skip_conv[idx](im_features[-idx - 2])
            x = self.blocks[idx](x)
            # print(f'x shape: {x.shape}')
            # output_features.append(x)
        # output_features.reverse()
        return x

class SecondEncoder(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., num_blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for num in range(num_blocks):
           
            if num == 0:
                self.downs.append(nn.Conv2d(in_channels= c * 2, out_channels= c, kernel_size=3, padding=1, stride=1, groups=1, bias=True))
            else:
                self.downs.append(nn.Conv2d(in_channels=c, out_channels=c * DW_Expand, kernel_size=3, padding=1, stride=2, groups=1, bias=True))
                c = c * DW_Expand
            self.blocks.append(BaselineBlock(c, DW_Expand, FFN_Expand, drop_out_rate))

    def forward(self, x):
        features = []
        for block, down in zip(self.blocks, self.downs):
            x = down(x)
            x = block(x)
            features.append(x)
        return x, features


class SecondDecoder(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.skip_conv = nn.ModuleList()
        self.num_blocks = num_blocks

        for num in range(num_blocks):
            self.ups.append(nn.ConvTranspose2d(in_channels=c, out_channels=c // DW_Expand, kernel_size=2, stride=2, bias=True))
            c = c // DW_Expand
            self.skip_conv.append(Event_Guided_Deformable_Conv(c, kernel_size=3))
            self.blocks.append(BaselineBlock(c, DW_Expand, FFN_Expand, drop_out_rate))
            
        

    def forward(self, x, im_features, ev_features):
        for idx in range(self.num_blocks):
            x = self.ups[idx](x)
            # print(f'im_features shape: {im_features[-idx - 2].shape}, ev_features shape: {ev_features[-idx - 2].shape}')
            x = x +self.skip_conv[idx](im_features[-idx - 2], ev_features[-idx - 2])
            x = self.blocks[idx](x)
        return x
    

class BaselineModel(nn.Module):
    def __init__(self, c = 32, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., event_channel = 128, num_blocks=3):
        super().__init__()
        self.ImgEncoder = ImgEncoder(c, DW_Expand, FFN_Expand, drop_out_rate, num_blocks)
        self.EventEncoder = EventEncoder(c, event_channel, DW_Expand, FFN_Expand, drop_out_rate, num_blocks)
        self.Decoder_1 = FirstDecoder(c * (2 ** (num_blocks - 1)), DW_Expand, FFN_Expand, drop_out_rate)
        self.Decoder_2 = SecondDecoder(c * (2 ** (num_blocks - 1)), DW_Expand, FFN_Expand, drop_out_rate)
        self.Encoder_2 = SecondEncoder(c, DW_Expand, FFN_Expand, drop_out_rate)
        self.SAM = SAM(c)
        self.concat = nn.Conv2d(in_channels=c * 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv = nn.Conv2d(in_channels= 3, out_channels=c, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.out_1 = nn.Conv2d(in_channels= c, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.out_2 = nn.Conv2d(in_channels=c, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
    def forward(self, img, event):
        ev_features = self.EventEncoder(event)
        x_img, im_features = self.ImgEncoder(img, ev_features)
        # for im in im_features:
            # print(f'Image features shape: {im.shape}')
        x = self.Decoder_1(x_img, im_features)
        # print(f'x shape: {x.shape}')
        sam_feat, out_1 = self.SAM(x, img)
        out_1 = img + out_1

        x, features = self.Encoder_2(torch.concat([self.conv(img), sam_feat], dim=1))

        # for im in features:
        #     print(f'Image features shape: {im.shape}')
        # print(f'x shape: {x.shape}')
        x = self.Decoder_2(x, features, ev_features)

        out_2 = img + self.out_2(x)
        return [out_1, out_2]
    