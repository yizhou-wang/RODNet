import torch.nn as nn

from .backbones.cdc import RadarVanilla
from .modules.mnet import MNet

try:
    from ..ops.dcn import DeformConvPack3D
except:
    print("Warning: DCN modules are not correctly imported!")


class RODNetCDCDCN(nn.Module):
    def __init__(self, in_channels, n_class, mnet_cfg=None, dcn=True):
        super(RODNetCDCDCN, self).__init__()
        self.dcn = dcn
        if dcn:
            self.conv_op = DeformConvPack3D
        else:
            self.conv_op = nn.Conv3d
        if mnet_cfg is not None:
            in_chirps_mnet, out_channels_mnet = mnet_cfg
            assert in_channels == in_chirps_mnet
            self.mnet = MNet(in_chirps_mnet, out_channels_mnet, conv_op=self.conv_op)
            self.with_mnet = True
            self.cdc = RadarVanilla(out_channels_mnet, n_class, use_mse_loss=False)
        else:
            self.with_mnet = False
            self.cdc = RadarVanilla(in_channels, n_class, use_mse_loss=False)

    def forward(self, x):
        if self.with_mnet:
            x = self.mnet(x)
        x = self.cdc(x)
        return x
