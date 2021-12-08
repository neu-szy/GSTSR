import torch
from torch import nn as nn

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import Upsample, make_layer


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True), nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0), nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class RCAB(nn.Module):
    """Residual Channel Attention Block (RCAB) used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, squeeze_factor=16, res_scale=1):
        super(RCAB, self).__init__()
        self.res_scale = res_scale

        self.rcab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.ReLU(True), nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor))

    def forward(self, x):
        res = self.rcab(x) * self.res_scale
        return res + x


class ResidualGroup(nn.Module):
    """Residual Group of RCAB.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_block (int): Block number in the body network.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, num_block, squeeze_factor=16, res_scale=1):
        super(ResidualGroup, self).__init__()

        self.residual_group = make_layer(
            RCAB, num_block, num_feat=num_feat, squeeze_factor=squeeze_factor, res_scale=res_scale)
        self.conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, x):
        res = self.conv(self.residual_group(x))
        return res + x

class LAM(nn.Module):
    def __init__(self, num_feat, num_group):
        super(LAM, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.conv1 = nn.Conv2d(num_feat * num_group, num_feat, 1, 1)
    def forward(self, x):
        n, c, h, w = x.shape
        batch = 1
        res = x
        x = x.view(batch, n, -1)
        res1 = x
        x = torch.bmm(x, x.transpose(1, 2))
        correlation_matrix = self.softmax(x)
        x = torch.bmm(correlation_matrix, res1)
        x = x.view(batch, n, c, h, w)
        x += res
        x = x.squeeze(0)
        x = self.conv1(x)
        return x

class CSAM(nn.Module):
    def __init__(self):
        super(CSAM, self).__init__()
        self.conv3 = nn.Conv3d(1, 1, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = x.unsqueeze(1)
        res_up = self.sigmoid(self.conv3(x))
        res_down = x
        x = res_up * x
        x = res_down + x
        x = x.squeeze(1)
        return x

@ARCH_REGISTRY.register()
# 用了这个修饰器会把“RCAN”添加到_obj_map中
class MY_HAN(nn.Module):
    """Residual Channel Attention Networks.

    Paper: Image Super-Resolution Using Very Deep Residual Channel Attention
        Networks
    Ref git repo: https://github.com/yulunzhang/RCAN.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_group (int): Number of ResidualGroup. Default: 10.
        num_block (int): Number of RCAB in ResidualGroup. Default: 16.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_group=10,
                 num_block=16,
                 squeeze_factor=16,
                 upscale=4,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(MY_HAN, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        self.rgs = nn.Sequential(*[ResidualGroup(num_feat=num_feat,
            num_block=num_block,
            squeeze_factor=squeeze_factor,
            res_scale=res_scale) for _ in range(num_group)])
        self.lam = LAM(num_feat, num_group)
        self.csam = CSAM()
        # self.body = make_layer(
        #     ResidualGroup,
        #     num_group,
        #     num_feat=num_feat,
        #     num_block=num_block,
        #     squeeze_factor=squeeze_factor,
        #     res_scale=res_scale)

        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        feature_group = []
        for rg in self.rgs:
            x = rg(x)
            feature_group.append(x)
        fg = torch.cat(feature_group, 1)
        la = self.lam(fg)
        res = self.conv_after_body(x)
        res = self.csam(res)
        res += la
        res += x

        x = self.conv_last(self.upsample(res))
        x = x / self.img_range + self.mean

        return x

if __name__ == "__main__":
    model = MY_HAN(3, 3)
    x = torch.rand((1, 3, 512, 512))
    model(x)
