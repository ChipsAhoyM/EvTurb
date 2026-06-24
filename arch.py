import torch.nn as nn
import torch


class GlobalAvgPool(nn.Module):
    """(N,C,H,W) -> (N,C)"""

    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        N, C, _, _ = x.shape
        return x.view(N, C, -1).mean(-1)


class SEBlock(nn.Module):
    """(N,C,H,W) -> (N,C,H,W)"""
    def __init__(self, in_channel, r):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            GlobalAvgPool(),
            nn.Linear(in_channel, in_channel // r),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(in_channel // r, in_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        se_weight = self.se(x).unsqueeze(-1).unsqueeze(-1)  # (N, C, 1, 1)
        return x * se_weight  # (N, C, H, W)

class Conv2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super(Conv2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class DeConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, pad=(0, 0)):
        super(DeConv2D, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2, output_padding=pad),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, input):
        return self.deconv(input)


class Dense_block(nn.Module):
    def __init__(self, in_ch, k):
        super(Dense_block, self).__init__()
        self.conv2d1_1 = Conv2D(in_ch, in_ch, 1, padding=0)
        self.conv2d1_2 = Conv2D(in_ch, k)

        self.conv2d2_1 = Conv2D(in_ch + k, in_ch, 1, padding=0)
        self.conv2d2_2 = Conv2D(in_ch, k)

        self.conv2d3_1 = Conv2D(in_ch + 2 * k, in_ch, 1, padding=0)
        self.conv2d3_2 = Conv2D(in_ch, k)

        self.conv2d4_1 = Conv2D(in_ch + 3 * k, in_ch, 1, padding=0)
        self.conv2d4_2 = Conv2D(in_ch, k)

    def forward(self, input):
        conv1_1 = self.conv2d1_1(input)
        conv1_2 = self.conv2d1_2(conv1_1)

        merge_1 = torch.cat([input, conv1_2], dim=1)
        conv2_1 = self.conv2d2_1(merge_1)
        conv2_2 = self.conv2d2_2(conv2_1)

        merge_2 = torch.cat([input, conv1_2, conv2_2], dim=1)
        conv3_1 = self.conv2d3_1(merge_2)
        conv3_2 = self.conv2d3_2(conv3_1)

        merge_3 = torch.cat([input, conv1_2, conv2_2, conv3_2], dim=1)
        conv4_1 = self.conv2d4_1(merge_3)
        conv4_2 = self.conv2d4_2(conv4_1)

        merge_4 = torch.cat([input, conv1_2, conv2_2, conv3_2, conv4_2], dim=1)
        return merge_4


class ResidualBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResidualBlock, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, 3, padding=1),
            nn.BatchNorm2d(channel_num),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, 3, padding=1),
            nn.BatchNorm2d(channel_num),
        )
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x + residual
        out = self.relu(x)
        return out