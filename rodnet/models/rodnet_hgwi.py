import torch
import torch.nn as nn

from .backbones.hgwi import RadarStackedHourglass


class RODNetHGwI(nn.Module):
    def __init__(self, n_class, stacked_num=1):
        super(RODNetHGwI, self).__init__()
        self.stacked_hourglass = RadarStackedHourglass(n_class, stacked_num=stacked_num)

    def forward(self, x):
        out = self.stacked_hourglass(x)
        return out


if __name__ == '__main__':
    testModel = RODNetHGwI().cuda()
    x = torch.zeros((1, 2, 16, 128, 128)).cuda()
    testModel(x)
