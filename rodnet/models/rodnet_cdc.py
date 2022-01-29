import torch.nn as nn

from .backbones.cdc import RadarVanilla


class RODNetCDC(nn.Module):
    def __init__(self, in_channels, n_class):
        super(RODNetCDC, self).__init__()
        self.cdc = RadarVanilla(in_channels, n_class, use_mse_loss=False)

    def forward(self, x):
        x = self.cdc(x)
        return x
