import torch.nn as nn

from .backbones.hg import RadarStackedHourglass


class RODNetHG(nn.Module):
    def __init__(self, in_channels, n_class, stacked_num=2):
        super(RODNetHG, self).__init__()
        self.stacked_hourglass = RadarStackedHourglass(in_channels, n_class, stacked_num=stacked_num)

    def forward(self, x):
        out = self.stacked_hourglass(x)
        return out
