import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


class ResBlock(nn.Module):
    def __init__(self, dim, use_dropout=False):
        super(ResBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, use_dropout)

    def build_conv_block(self, dim, use_dropout):
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
                       nn.InstanceNorm2d(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


## https://github.com/swz30/MPRNet
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = torch.nn.Conv2d(n_feat, n_feat, kernel_size, 1, kernel_size//2, bias=bias)
        self.conv2 = torch.nn.Conv2d(n_feat, 3, kernel_size, 1, kernel_size//2, bias=bias)
        self.conv3 = torch.nn.Conv2d(3, n_feat, kernel_size, 1, kernel_size//2, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


class LayerNorm2d(nn.Module):
    def __init__(self, num_features):
        super(LayerNorm2d, self).__init__()
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, -1).transpose(1, 2)
        x = self.layer_norm(x)
        x = x.transpose(1, 2).view(b, c, h, w)
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class BaselineBlock(nn.Module):
    def __init__(self, c, DW_Expand=1, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
            nn.Sigmoid()
        )

        self.gelu = nn.GELU()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = x * self.se(x)
        x = self.conv3(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.gelu(x)
        x = self.conv5(x)
        return y + x * self.gamma


class EventImageFusion(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion=2, bias=False):
        super(EventImageFusion, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.norm_image = LayerNorm2d(dim)
        self.norm_event = LayerNorm2d(dim)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.fc = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * ffn_expansion),
            nn.GELU(),
            nn.Linear(dim * ffn_expansion, dim)
        )

    def forward(self, image, event):
        assert image.size() == event.size(), f'Image size {image.size()} and event size {event.size()} should be the same'
        b, c, h, w = image.size()
        image = self.norm_image(image)
        event = self.norm_event(event)

        q = self.q(image)
        k = self.k(event)
        v = self.v(event)

        q = q.view(b, self.num_heads, c // self.num_heads, h, w).permute(0, 1, 3, 4, 2)
        k = k.view(b, self.num_heads, c // self.num_heads, h, w).permute(0, 1, 3, 4, 2)
        v = v.view(b, self.num_heads, c // self.num_heads, h, w).permute(0, 1, 3, 4, 2)

        attn = (q @ k.transpose(-2, -1)) / (c // self.num_heads) ** 0.5
        attn = F.softmax(attn, dim=-1)
        x = attn @ v
        x = x.permute(0, 1, 4, 2, 3).contiguous().view(b, c, h, w)
        x = self.fc(x)
        x = x + image

        x = x.view(b, h * w, c)
        x = x + self.mlp(self.norm2(x))
        x = x.view(b, c, h, w)
        return x


class Event_Guided_Deformable_Conv(nn.Module):
    """Event-Guided Deformable Convolution (EGDC) used in D-Net skip connections."""
    def __init__(self, channels, kernel_size=3, groups=1):
        super(Event_Guided_Deformable_Conv, self).__init__()
        dim = kernel_size * kernel_size * groups * 2
        self.proj_conv_structure = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
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
        x = deform_conv2d(x, offset=offset, mask=mask, padding=1, stride=1, dilation=1, weight=self.weight)
        return x
