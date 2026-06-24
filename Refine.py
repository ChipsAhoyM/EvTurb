from arch import *
from Blocks import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class SCAM(nn.Module):
    '''
    Stereo Cross Attention Module (SCAM)
    '''
    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5

        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x_l, x_r):
        Q_l = self.l_proj1(self.norm_l(x_l)).permute(0, 2, 3, 1)  # B, H, W, c
        Q_r_T = self.r_proj1(self.norm_r(x_r)).permute(0, 2, 1, 3) # B, H, c, W (transposed)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  #B, H, W, c
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l) #B, H, W, c

        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma
        return x_l + F_r2l, x_r + F_l2r

class NAFBlockSR(nn.Module):
    def __init__(self, c, vp_c):
        super().__init__()
        self.blk = ResidualBlock(c)
        self.conv = nn.Conv2d(in_channels=vp_c, out_channels=c, kernel_size=3, padding=1)
        self.fusion = SCAM(c)
        self.out_conv = nn.Conv2d(in_channels=c, out_channels=vp_c, kernel_size=3, padding=1)
        self.norm = LayerNorm2d(vp_c)
        

    def forward(self, feat, vp):
        feats = self.blk(feat)
        vp_f = self.conv(vp)
        feats, out_vp = self.fusion(feats, vp_f)
        out_vp = self.norm(self.out_conv(out_vp))
        return tuple([feats, out_vp])
    
class RefineNet(nn.Module):
    def __init__(self, width=48, num_blks=4, img_channel=3, vp_channel=1):
        super().__init__()
        
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1)
        self.body = MySequential(
            *[NAFBlockSR(width, vp_channel) for i in range(num_blks)]
        )

        self.con = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1)
        )

    def forward(self, img, vp):
        feats = tuple([self.intro(img), vp])
        feats = self.body(*feats)
        out = self.con(feats[0])
        out = out + img
        return out
