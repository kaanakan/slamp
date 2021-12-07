import torch
import torch.nn as nn
import torch.nn.functional as F


def activation_factory(name):
    """
    Returns the activation layer corresponding to the input activation name.
    Parameters
    ----------
    name : str
        'relu', 'leaky_relu', 'elu', 'sigmoid', or 'tanh'. Adds the corresponding activation function after the
        convolution.
    Returns
    -------
    torch.nn.Module
        Element-wise activation layer.
    """
    if name == 'relu':
        return nn.ReLU(inplace=True)
    if name == 'leaky_relu':
        return nn.LeakyReLU(0.2, inplace=True)
    if name == 'elu':
        return nn.ELU(inplace=True)
    if name == 'sigmoid':
        return nn.Sigmoid()
    if name == 'tanh':
        return nn.Tanh()
    raise ValueError(f'Activation function \'{name}\' not yet implemented')


class conv(nn.Module):
    """
    General convolutional layer
    """

    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1, act='relu', bn=False):
        super(conv, self).__init__()

        layers = [nn.Conv2d(int(in_channels), int(out_channels), 3, stride, padding, bias=not bn)]
        if bn:
            layers.append(nn.BatchNorm2d(int(out_channels)))
        layers.append(activation_factory(act))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class spatial_encoder(nn.Module):
    def __init__(self, dim, nc=1, bn=True):
        super(spatial_encoder, self).__init__()
        self.dim = dim
        self.nc = nc
        self.c1 = nn.Sequential(
            conv(nc, 256, act='leaky_relu', bn=bn),
            conv(256, 256, act='leaky_relu', bn=bn),
            SELayer(256),
        )
        self.c2 = nn.Sequential(
            conv(256, 256, act='leaky_relu', bn=bn),
            conv(256, 128, act='leaky_relu', bn=bn),
            SELayer(128),
        )
        self.c3 = nn.Sequential(
            conv(128, 64, act='leaky_relu', bn=bn),
            conv(64, dim, act='leaky_relu', bn=bn),
        )

    def forward(self, x):
        out = self.c1(x)
        out = self.c2(out)
        out = self.c3(out)
        return out


class encoder(nn.Module):
    def __init__(self, dim, nc=3, h=92, w=310, bn=True):
        super(encoder, self).__init__()
        self.dim = dim
        # 92x310
        self.c1 = nn.Sequential(
            conv(nc, 64, act='leaky_relu', kernel=3, stride=2, padding=1, bn=bn),
            conv(64, 64, act='leaky_relu', bn=bn)
        )
        # 32x32
        self.c2 = nn.Sequential(
            conv(64, 96, act='leaky_relu', kernel=3, stride=2, padding=1, bn=bn),
            conv(96, 96, act='leaky_relu', bn=bn)
        )
        # 16x16
        self.c3 = nn.Sequential(
            conv(96, 128, act='leaky_relu', stride=2, bn=bn),
            conv(128, 128, act='leaky_relu', bn=bn),
            conv(128, 128, act='leaky_relu', bn=bn)
        )
        # 8x8
        self.c4 = nn.Sequential(
            conv(128, 192, act='leaky_relu', stride=2, bn=bn),
            conv(192, 192, act='leaky_relu', bn=bn),
            conv(192, 192, act='leaky_relu', bn=bn)
        )
        self.c5 = nn.Sequential(
            conv(192, 256, act='leaky_relu', bn=bn),
            conv(256, 256, act='leaky_relu', bn=bn),
            conv(256, 256, act='leaky_relu', bn=bn)
        )
        # 4x4
        self.c6 = nn.Sequential(
            conv(256, dim, act='leaky_relu', bn=bn),
            conv(dim, dim, act='leaky_relu', bn=bn)
        )
        self.pools = [
            nn.AdaptiveMaxPool2d((46, 156)),
            nn.AdaptiveMaxPool2d((24, 80)),
            nn.AdaptiveMaxPool2d((12, 40)),
            nn.AdaptiveMaxPool2d((6, 20)),
            nn.AdaptiveMaxPool2d((4, 4)),
        ]

        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        h1 = self.pools[0](self.c1(x))  # 64x64
        h2 = self.pools[1](self.c2(h1))  # 32x32
        h3 = self.pools[2](self.c3(h2))  # 16x16
        h4 = self.pools[3](self.c4(h3))  # 8x8
        h5 = self.pools[4](self.c5(h4))  # 4x4
        h6 = self.c6(h5)
        return h6, [h1, h2, h3, h4, h5]


class decoder(nn.Module):
    def __init__(self, dim, nc=1, act=nn.Sigmoid, bn=True, num_scales=1):
        super(decoder, self).__init__()
        self.dim = dim
        self.act = act
        self.num_scales = num_scales

        self.upc1 = nn.Sequential(
            conv(dim, 256, act='leaky_relu', bn=True),
            conv(256, 256, act='leaky_relu', bn=True),
        )
        self.upc2 = nn.Sequential(
            conv(256 * 2, 256, act='leaky_relu', bn=bn),
            conv(256, 256, act='leaky_relu', bn=bn),
            conv(256, 192, act='leaky_relu', bn=bn)
        )
        self.upc3 = nn.Sequential(
            conv(192 * 2, 192, act='leaky_relu', bn=bn),
            conv(192, 192, act='leaky_relu', bn=bn),
            conv(192, 128, act='leaky_relu', bn=bn)
        )
        self.upc4 = nn.Sequential(
            conv(128 * 2, 128, act='leaky_relu', bn=bn),
            conv(128, 96, act='leaky_relu', bn=bn),
        )
        self.upc5 = nn.Sequential(
            conv(96 * 2, 96, act='leaky_relu', bn=bn),
            conv(96, 64, act='leaky_relu', bn=bn),
        )
        # 64 x 64
        self.upc6 = nn.Sequential(
            conv(64 * 2, 64, act='leaky_relu', bn=bn),
            conv(64, 64, act='leaky_relu', bn=bn),
        )
        if self.act:
            self.upc7 = nn.Sequential(
                conv(64, 64, act='leaky_relu', bn=bn),
                nn.ConvTranspose2d(64, nc, 3, 1, 1),
                nn.Sigmoid()
            )
        else:
            self.upc7 = nn.Sequential(
                conv(64, 64, act='leaky_relu', bn=bn),
                nn.ConvTranspose2d(64, nc, 3, 1, 1)
            )
        if self.num_scales == 3:
            self.conv1x1 = conv(96, 64, act='leaky_relu', bn=bn, kernel=1)

    def forward(self, inp):
        x, skips = inp
        d1 = self.upc1(x)  # 1 -> 4
        d2 = self.upc2(torch.cat([d1, skips[4]], 1))  # 8 x 8
        up2 = F.interpolate(d2, size=(6, 20), mode='bilinear', align_corners=True)  # self.up(d1) # 4 -> 8
        d3 = self.upc3(torch.cat([up2, skips[3]], 1))  # 16 x 16
        up3 = F.interpolate(d3, size=(12, 40), mode='bilinear', align_corners=True)  # self.up(d1) # 4 -> 8
        d4 = self.upc4(torch.cat([up3, skips[2]], 1))  # 32 x 32
        up4 = F.interpolate(d4, size=(24, 80), mode='bilinear', align_corners=True)  # self.up(d1) # 4 -> 8
        d5 = self.upc5(torch.cat([up4, skips[1]], 1))  # 64 x 64
        up5 = F.interpolate(d5, size=(46, 156), mode='bilinear', align_corners=True)  # self.up(d1) # 4 -> 8
        d6 = self.upc6(torch.cat([up5, skips[0]], 1))
        up6 = F.interpolate(d6, size=(92, 310), mode='bilinear', align_corners=True)  # self.up(d1) # 4 -> 8
        out = self.upc7(up6)
        return out
