import torch.nn as nn
from torch import tanh

# Firt convolutional block of residual block
class FirstConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, discriminator=False):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=True)
        if  not discriminator:
            self.normalization = nn.Identity()
            self.activation = nn.PReLU(num_parameters=out_channels)
        else:
            self.normalization = nn.InstanceNorm2d(out_channels)
            self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        i = self.cnn(x)
        i = self.normalization(i)
        i = self.activation(i)
        return i

# Second convolutional block of residual block
class SecondConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=True)
        self.normalization = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        i = self.cnn(x)
        i = self.normalization(i)
        return i

# Residual Block
class GeneratorResidual(nn.Module):
    def __init__(self, in_channels, kernel, stride, padding):
        super().__init__()
        """
        seq = []
        seq.append(FirstConv(in_channels=in_channels,
                                      out_channels=in_channels,
                                      kernel, stride, padding))
        seq.append(SecondConv(in_channels=in_channels,
                                      out_channels=in_channels,
                                      kernel, stride, padding))
        self.seq = nn.Sequential(*seq)
        """
        self.first = FirstConv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel=kernel, stride=stride, padding=padding
        )
        self.second = SecondConv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel=kernel, stride=stride, padding=padding
        )

    def forward(self, x):
        # i = self.seq(x)
        x1 = self.first(x)
        x1_ = x + x1
        x2 = self.second(x1_)
        return x + x1 + x2

# Upsample convolution
class Upsample(nn.Module):
    def __init__(self, in_channels, scale):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * (scale ** 2), kernel_size=3, stride=1, padding=1)
        self.shuffle = nn.PixelShuffle(scale)
        self.activation = nn.PReLU(num_parameters=in_channels)

    def forward(self, x):
        i = self.conv(x)
        i = self.shuffle(i)
        i = self.activation(i)
        return i

class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_residual=5, scale=4):
        super().__init__()
        blocks = []
        self.initial_conv = nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=9, stride=1, padding=9//2, bias=False)
        self.initial_act = nn.PReLU(num_parameters=num_channels)
        self.residual = nn.Sequential(*[GeneratorResidual(num_channels, kernel=3, stride=1, padding=1) for _ in range(num_residual)])
        self.mid_conv = SecondConv(num_channels, num_channels, kernel=9, stride=1, padding=9//2)
        self.upsample = nn.Sequential(*[Upsample(num_channels, scale=scale//2) for _ in range(2)])
        self.last_conv = nn.Conv2d(in_channels=num_channels, out_channels=in_channels, kernel_size=9, stride=1, padding=9//2)
        blocks = []

    def forward(self, x):
        i = self.initial_conv(x)
        i = self.initial_act(i)
        o = self.residual(i)
        o = self.mid_conv(o) + i
        o = self.upsample(o)
        o = self.last_conv(o)
        o = tanh(o)
        return o