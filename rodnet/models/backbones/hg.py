import torch.nn as nn


class RadarStackedHourglass(nn.Module):

    def __init__(self, in_channels, n_class, stacked_num=1, conv_op=None, use_mse_loss=False):
        super(RadarStackedHourglass, self).__init__()
        self.stacked_num = stacked_num
        if conv_op is None:
            self.conv1a = nn.Conv3d(in_channels=in_channels, out_channels=32,
                                    kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        else:
            self.conv1a = conv_op(in_channels=in_channels, out_channels=32,
                                  kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1))

        self.conv1b = nn.Conv3d(in_channels=32, out_channels=64,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))

        self.hourglass = []
        for i in range(stacked_num):
            self.hourglass.append(nn.ModuleList([RODEncode(), RODDecode(),
                                                 nn.Conv3d(in_channels=64, out_channels=n_class,
                                                           kernel_size=(9, 5, 5), stride=(1, 1, 1),
                                                           padding=(4, 2, 2)),
                                                 nn.Conv3d(in_channels=n_class, out_channels=64,
                                                           kernel_size=(9, 5, 5), stride=(1, 1, 1),
                                                           padding=(4, 2, 2))]))
        self.hourglass = nn.ModuleList(self.hourglass)
        self.relu = nn.ReLU()
        self.bn1a = nn.BatchNorm3d(num_features=32)
        self.bn1b = nn.BatchNorm3d(num_features=64)
        self.sigmoid = nn.Sigmoid()
        self.use_mse_loss = use_mse_loss

    def forward(self, x):
        x = self.relu(self.bn1a(self.conv1a(x)))
        x = self.relu(self.bn1b(self.conv1b(x)))

        out = []
        for i in range(self.stacked_num):
            x, x1, x2, x3 = self.hourglass[i][0](x)
            x = self.hourglass[i][1](x, x1, x2, x3)
            confmap = self.hourglass[i][2](x)
            if not self.use_mse_loss:
                confmap = self.sigmoid(confmap)
            out.append(confmap)
            if i < self.stacked_num - 1:
                confmap_ = self.hourglass[i][3](confmap)
                x = x + confmap_
        return out


class RODEncode(nn.Module):

    def __init__(self):
        super(RODEncode, self).__init__()
        self.conv1a = nn.Conv3d(in_channels=64, out_channels=64,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv1b = nn.Conv3d(in_channels=64, out_channels=64,
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.conv2a = nn.Conv3d(in_channels=64, out_channels=128,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv2b = nn.Conv3d(in_channels=128, out_channels=128,
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.conv3a = nn.Conv3d(in_channels=128, out_channels=256,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv3b = nn.Conv3d(in_channels=256, out_channels=256,
                                kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))

        self.skipconv1a = nn.Conv3d(in_channels=64, out_channels=64,
                                    kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.skipconv1b = nn.Conv3d(in_channels=64, out_channels=64,
                                    kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.skipconv2a = nn.Conv3d(in_channels=64, out_channels=128,
                                    kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.skipconv2b = nn.Conv3d(in_channels=128, out_channels=128,
                                    kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.skipconv3a = nn.Conv3d(in_channels=128, out_channels=256,
                                    kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.skipconv3b = nn.Conv3d(in_channels=256, out_channels=256,
                                    kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))
        self.bn1a = nn.BatchNorm3d(num_features=64)
        self.bn1b = nn.BatchNorm3d(num_features=64)
        self.bn2a = nn.BatchNorm3d(num_features=128)
        self.bn2b = nn.BatchNorm3d(num_features=128)
        self.bn3a = nn.BatchNorm3d(num_features=256)
        self.bn3b = nn.BatchNorm3d(num_features=256)

        self.skipbn1a = nn.BatchNorm3d(num_features=64)
        self.skipbn1b = nn.BatchNorm3d(num_features=64)
        self.skipbn2a = nn.BatchNorm3d(num_features=128)
        self.skipbn2b = nn.BatchNorm3d(num_features=128)
        self.skipbn3a = nn.BatchNorm3d(num_features=256)
        self.skipbn3b = nn.BatchNorm3d(num_features=256)

        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.skipbn1a(self.skipconv1a(x)))
        x1 = self.relu(self.skipbn1b(self.skipconv1b(x1)))
        x = self.relu(self.bn1a(self.conv1a(x)))  # (B, 2, W, 128, 128) -> (B, 64, W, 128, 128)
        x = self.relu(self.bn1b(self.conv1b(x)))  # (B, 64, W, 128, 128) -> (B, 64, W/2, 64, 64)

        x2 = self.relu(self.skipbn2a(self.skipconv2a(x)))
        x2 = self.relu(self.skipbn2b(self.skipconv2b(x2)))
        x = self.relu(self.bn2a(self.conv2a(x)))  # (B, 64, W/2, 64, 64) -> (B, 128, W/2, 64, 64)
        x = self.relu(self.bn2b(self.conv2b(x)))  # (B, 128, W/2, 64, 64) -> (B, 128, W/4, 32, 32)

        x3 = self.relu(self.skipbn3a(self.skipconv3a(x)))
        x3 = self.relu(self.skipbn3b(self.skipconv3b(x3)))
        x = self.relu(self.bn3a(self.conv3a(x)))  # (B, 128, W/4, 32, 32) -> (B, 256, W/4, 32, 32)
        x = self.relu(self.bn3b(self.conv3b(x)))  # (B, 256, W/4, 32, 32) -> (B, 256, W/4, 16, 16)

        return x, x1, x2, x3


class RODDecode(nn.Module):

    def __init__(self):
        super(RODDecode, self).__init__()
        self.convt1 = nn.ConvTranspose3d(in_channels=256, out_channels=128,
                                         kernel_size=(3, 6, 6), stride=(1, 2, 2), padding=(1, 2, 2))
        self.convt2 = nn.ConvTranspose3d(in_channels=128, out_channels=64,
                                         kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.convt3 = nn.ConvTranspose3d(in_channels=64, out_channels=64,
                                         kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        # self.upsample = nn.Upsample(size=(rodnet_configs['win_size'], radar_configs['ramap_rsize'],
        #                                   radar_configs['ramap_asize']), mode='nearest')

    def forward(self, x, x1, x2, x3):
        x = self.prelu(self.convt1(x + x3))  # (B, 256, W/4, 16, 16) -> (B, 128, W/2, 32, 32)
        x = self.prelu(self.convt2(x + x2))  # (B, 128, W/2, 32, 32) -> (B, 64, W, 64, 64)
        x = self.convt3(x + x1)  # (B, 64, W, 64, 64) -> (B, 3, W, 128, 128)
        return x
