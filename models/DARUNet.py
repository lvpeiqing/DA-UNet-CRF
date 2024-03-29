import torch
import torch.nn as nn
import torch.nn.functional as F
import parameter as para


#### Inception Redidual Net

class IncResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(IncResBlock, self).__init__()
        self.Inputconv1x1 = nn.Conv3d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        #
        self.conv1_1 = nn.Sequential(
            nn.Conv3d(inplanes, planes // 4, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm3d(planes // 4))
        # nn.ReLU())
        self.conv1_2 = nn.Sequential(
            nn.Conv3d(inplanes, planes // 4, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm3d(planes // 4),
            nn.PReLU(),
            nn.Conv3d(planes // 4, planes // 4, kernel_size=3, stride=stride, dilation=2, padding=2, bias=False),
            nn.BatchNorm3d(planes // 4))
        self.conv1_3 = nn.Sequential(
            nn.Conv3d(inplanes, planes // 4, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm3d(planes // 4),
            nn.ReLU(),
            nn.Conv3d(planes // 4, planes // 4, kernel_size=3, stride=stride, dilation=4, padding=4, bias=False),
            nn.BatchNorm3d(planes // 4))
        self.conv1_4 = nn.Sequential(
            nn.Conv3d(inplanes, planes // 4, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm3d(planes // 4),
            nn.ReLU(),
            nn.Conv3d(planes // 4, planes // 4, kernel_size=3, stride=stride, dilation=8, padding=8, bias=False),
            nn.BatchNorm3d(planes // 4))
        self.relu = nn.ReLU()
        self.stride = stride

    def forward(self, x):
        residual = self.Inputconv1x1(x)

        c1 = self.conv1_1(x)
        c2 = self.conv1_2(x)
        c3 = self.conv1_3(x)
        c4 = self.conv1_4(x)

        out = torch.cat([c1, c2, c3, c4], 1)

        # adding the skip connection
        out += residual
        out = self.relu(out)

        return out


class DARUNet(nn.Module):
    """

    共9498260个可训练的参数, 接近九百五十万
    """
    def __init__(self, training):
        super().__init__()

        self.training = training

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(1, 16, 3, 1, padding=1),
            nn.PReLU(16),

            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1, ),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1, ),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=1, ),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1, ),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1, ),
            nn.PReLU(128),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256),

            IncResBlock(256,256),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.PReLU(128)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.PReLU(128)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.PReLU(64)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.PReLU(32)
        )

        # 最后大尺度下的映射（256*256），下面的尺度依次递减
        self.map4 = nn.Sequential(
            nn.Conv3d(32, 1, 1, 1),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear',align_corners=True),
            nn.Sigmoid()
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(64, 1, 1, 1),
            nn.Upsample(scale_factor=(2, 4, 4), mode='trilinear',align_corners=True),
            nn.Sigmoid()
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(128, 1, 1, 1),
            nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear',align_corners=True),
            nn.Sigmoid()
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, 1, 1, 1),
            nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear',align_corners=True),
            nn.Sigmoid()
        )

    def forward(self, inputs):

        long_range1 = self.encoder_stage1(inputs) + inputs

        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1) + short_range1
        long_range2 = F.dropout(long_range2, para.drop_rate, self.training)

        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2) + short_range2
        long_range3 = F.dropout(long_range3, para.drop_rate, self.training)

        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.encoder_stage4(short_range3) + short_range3
        long_range4 = F.dropout(long_range4, para.drop_rate, self.training)

        short_range4 = self.down_conv4(long_range4)

        outputs = self.decoder_stage1(long_range4) + short_range4
        outputs = F.dropout(outputs, para.drop_rate, self.training)

        output1 = self.map1(outputs)

        short_range6 = self.up_conv2(outputs)

        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        outputs = F.dropout(outputs, 0.3, self.training)

        output2 = self.map2(outputs)

        short_range7 = self.up_conv3(outputs)

        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        outputs = F.dropout(outputs, 0.3, self.training)

        output3 = self.map3(outputs)

        short_range8 = self.up_conv4(outputs)

        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8

        output4 = self.map4(outputs)

        # outputs = output1 + output2 + output3 + output4

        if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4


def init(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(module.weight.data, 0.25)
        nn.init.constant_(module.bias.data, 0)