import math
import torch
import torch.nn as nn


class MNet(nn.Module):
    def __init__(self, in_chirps, out_channels, conv_op=None):
        super(MNet, self).__init__()
        self.in_chirps = in_chirps
        self.out_channels = out_channels
        if conv_op is None:
            conv_op = nn.Conv3d
        self.conv_op = conv_op

        self.t_conv3d = conv_op(in_channels=2, out_channels=out_channels, kernel_size=(3, 1, 1), stride=(2, 1, 1),
                                padding=(1, 0, 0))
        t_conv_out = math.floor((in_chirps + 2 * 1 - (3 - 1) - 1) / 2 + 1)
        self.t_maxpool = nn.MaxPool3d(kernel_size=(t_conv_out, 1, 1))

    def forward(self, x):
        batch_size, n_channels, win_size, in_chirps, w, h = x.shape
        x_out = torch.zeros((batch_size, self.out_channels, win_size, w, h)).cuda()
        for win in range(win_size):
            x_win = self.t_conv3d(x[:, :, win, :, :, :])
            x_win = self.t_maxpool(x_win)
            x_win = x_win.view(batch_size, self.out_channels, w, h)
            x_out[:, :, win, ] = x_win
        return x_out


if __name__ == '__main__':
    batch_size = 4
    in_channels = 2
    win_size = 32
    in_chirps = 4
    w = 128
    h = 128
    out_channels = 32
    mnet = MNet(in_chirps=in_chirps, out_channels=out_channels)
    input = torch.randn(batch_size, in_channels, win_size, in_chirps, w, h)
    output = mnet(input)
