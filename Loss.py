import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class HybridLoss(nn.Module):
    def __init__(self, perceptual_weight=1.0, tv_weight=1.0):
        super(HybridLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.perceptual_weight = perceptual_weight
        self.tv_weight = tv_weight

        vgg = models.vgg16(pretrained=True).features
        self.vgg_layers = nn.Sequential(*list(vgg)[:16]).eval()
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        mse_loss = self.mse_loss(input, target)

        input_features = self.vgg_layers(input)
        target_features = self.vgg_layers(target)
        perceptual_loss = F.mse_loss(input_features, target_features)
        tv_loss = self.tv_loss(input)
        total_loss = mse_loss + self.perceptual_weight * perceptual_loss + self.tv_weight * tv_loss
        return total_loss

    def tv_loss(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h_x - 1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w_x - 1], 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]
