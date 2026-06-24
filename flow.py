import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision.utils import flow_to_image

from warper import warp
from Blocks import OverlapPatchEmbed, BaselineBlock


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        nn.PReLU(out_planes)
    )

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


def vis_flow(flow, filename):
    img = flow_to_image(flow)
    img = img.squeeze().cpu().permute(1, 2, 0).numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)
    return img


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            conv(c, c), conv(c, c), conv(c, c), conv(c, c),
            conv(c, c), conv(c, c), conv(c, c), conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 2, 4, 2, 1)

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor=scale * 2, mode="bilinear", align_corners=False)
        flow = tmp[:, :2] * scale * 2
        return flow


class IFNet(nn.Module):
    """T-Net: variance-map guided tilt flow estimation for turbulence correction."""
    def __init__(self):
        super(IFNet, self).__init__()
        self.patch_embed1 = OverlapPatchEmbed(3, 48)
        self.patch_embed2 = OverlapPatchEmbed(1, 48)
        self.merge_conv = nn.Conv2d(48 * 2, 48, kernel_size=1, bias=False)
        self.encoder = nn.Sequential(*[BaselineBlock(c=48) for _ in range(3)])

        self.block0 = IFBlock(48, c=240)
        self.block1 = IFBlock(48 + 2, c=150)
        self.block2 = IFBlock(48 + 2, c=90)

    def forward(self, img, vp, scale=[4, 2, 1], save_flow=False, batchidx=None):
        flow = None
        flow_list = []
        stu = [self.block0, self.block1, self.block2]

        inp_enc = self.patch_embed1(img)
        vp_enc = self.patch_embed2(vp)
        merged = self.merge_conv(torch.cat([inp_enc, vp_enc], dim=1))
        out_enc = self.encoder(merged)

        for i in range(3):
            if flow is not None:
                flow_d = stu[i](out_enc, flow, scale=scale[i])
                flow = flow + flow_d
            else:
                flow = stu[i](out_enc, None, scale=scale[i])
            flow_list.append(flow)

        warped_img = warp(img, flow)

        if save_flow:
            vis_flow(flow_list[0], f'flow{batchidx}_1.png')
            vis_flow(flow_list[1], f'flow{batchidx}_2.png')
            vis_flow(flow_list[2], f'flow{batchidx}_3.png')

        return warped_img


if __name__ == '__main__':
    model = IFNet().cuda()
    img = torch.randn(1, 3, 256, 256).cuda()
    vp = torch.randn(1, 1, 256, 256).cuda()
    out = model(img, vp)
    print(out.shape)
