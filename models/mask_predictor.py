import torch.nn as nn


class vgg_layer(nn.Module):
    def __init__(self, nin, nout):
        super(vgg_layer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nin, nout, 3, 1, 1, bias=False),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        return self.main(input)


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


class mask_predictor(nn.Module):
    def __init__(self, nc=1):
        super(mask_predictor, self).__init__()
        # 64 x 64
        self.c1 = nn.Sequential(
            vgg_layer(nc, 64),
            vgg_layer(64, 64),
            SELayer(64),
        )
        # 32 x 32
        self.c2 = nn.Sequential(
            vgg_layer(64, 64),
            vgg_layer(64, 64),
            SELayer(64),
        )
        # 16 x 16 
        # 4 x 4
        self.c3 = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        h1 = self.c1(input)  # 64 -> 32
        h2 = self.c2((h1))  # 32 -> 16
        h3 = self.c3((h2))  # 16 -> 8
        return h3
