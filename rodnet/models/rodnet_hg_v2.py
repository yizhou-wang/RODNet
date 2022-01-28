import torch
import torch.nn as nn

from .backbones.hg_dcn import RadarStackedHourglass
from .prep_layers.mnet import MNet
from ..ops.dcn import DeformConvPack3D


class RODNetHGDCNv3(nn.Module):
    def __init__(self, n_class, stacked_num=2, mnet_cfg=None):
        super(RODNetHGDCNv3, self).__init__()
        if mnet_cfg is not None:
            in_chirps_mnet, out_channels_mnet = mnet_cfg
            self.mnet = MNet(in_chirps_mnet, out_channels_mnet, conv_op=DeformConvPack3D)
            self.with_mnet = True
            self.stacked_hourglass = RadarStackedHourglass(n_class, stacked_num=stacked_num,
                                                           in_channels=out_channels_mnet, conv_op=DeformConvPack3D)
        else:
            self.with_mnet = False
            self.stacked_hourglass = RadarStackedHourglass(n_class, stacked_num=stacked_num, conv_op=DeformConvPack3D)

    def forward(self, x):
        if self.with_mnet:
            x = self.mnet(x)
        out, offsets = self.stacked_hourglass(x)
        return out, offsets


if __name__ == '__main__':
    torch.cuda.set_device(torch.device('cuda:0'))
    batch_size = 1
    in_channels = 2
    win_size = 4
    in_chirps = 6
    w = 64
    h = 64
    out_channels = 8
    model = RODNetHGDCN(n_class=3, stacked_num=1, mnet_cfg=(in_chirps, out_channels)).cuda()
    for iter in range(10):
        input = torch.randn(batch_size, in_channels, win_size, in_chirps, w, h).cuda()
        output = model(input)
        print("forward done")
        output_gt = torch.randn(batch_size, 3, win_size, w, h).cuda()
        criterion = nn.BCELoss()
        loss = criterion(output[0], output_gt)
        loss.backward()

