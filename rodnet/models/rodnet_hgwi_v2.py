import torch
import torch.nn as nn

from .backbones.hgwi import RadarStackedHourglass
from .modules.mnet import MNet

try:
    from ..ops.dcn import DeformConvPack3D
except:
    print("Warning: DCN modules are not correctly imported!")


class RODNetHGwIDCN(nn.Module):
    def __init__(self, in_channels, n_class, stacked_num=1, mnet_cfg=None, dcn=True):
        super(RODNetHGwIDCN, self).__init__()
        self.dcn = dcn
        if dcn:
            self.conv_op = DeformConvPack3D
        else:
            self.conv_op = nn.Conv3d
        if mnet_cfg is not None:
            in_chirps_mnet, out_channels_mnet = mnet_cfg
            self.mnet = MNet(in_chirps_mnet, out_channels_mnet, conv_op=self.conv_op)
            self.with_mnet = True
            self.stacked_hourglass = RadarStackedHourglass(out_channels_mnet, n_class, stacked_num=stacked_num,
                                                           conv_op=self.conv_op)
        else:
            self.with_mnet = False
            self.stacked_hourglass = RadarStackedHourglass(in_channels, n_class, stacked_num=stacked_num,
                                                           conv_op=self.conv_op)

    def forward(self, x):
        if self.with_mnet:
            x = self.mnet(x)
        out = self.stacked_hourglass(x)
        return out


if __name__ == '__main__':
    testModel = RODNetHGwIDCN().cuda()
    x = torch.zeros((1, 2, 16, 128, 128)).cuda()
    testModel(x)
