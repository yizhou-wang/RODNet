import torch
import torch.nn as nn

from .backbones.hg import RadarStackedHourglass
from .modules.mnet import MNet

try:
    from ..ops.dcn import DeformConvPack3D
except:
    print("Warning: DCN modules are not correctly imported!")


class RODNetHGDCN(nn.Module):
    def __init__(self, in_channels, n_class, stacked_num=2, mnet_cfg=None, dcn=True):
        super(RODNetHGDCN, self).__init__()
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
