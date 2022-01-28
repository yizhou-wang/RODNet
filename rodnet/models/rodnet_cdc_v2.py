import torch.nn as nn

from .backbones.cdc import RODEncode, RODDecode
from .modules.mnet import MNet
from ..ops.dcn import DeformConvPack3D


class RODNetCDCDCN(nn.Module):
    def __init__(self, n_class, mnet_cfg=None):
        super(RODNetCDCDCN, self).__init__()
        if mnet_cfg is not None:
            in_chirps_mnet, out_channels_mnet = mnet_cfg
            self.mnet = MNet(in_chirps_mnet, out_channels_mnet, conv_op=DeformConvPack3D)
            self.with_mnet = True
            self.c3d_encode = RODEncode(in_channels=out_channels_mnet)
        else:
            self.with_mnet = False
            self.c3d_encode = RODEncode()
        self.c3d_decode = RODDecode(n_class)

    def forward(self, x):
        if self.with_mnet:
            x = self.mnet(x)
        x = self.c3d_encode(x)
        dets = self.c3d_decode(x)
        return dets
