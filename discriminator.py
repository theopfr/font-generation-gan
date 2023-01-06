import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, 
                channels_in: int, 
                channels_out: int, 
                kernel_size: int,
                stride: int,
                padding: int,
                norm: bool=True,
                last: bool=False,
        ):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding, bias=False)
        self.batchnorm = nn.BatchNorm2d(channels_out) if norm else nn.Identity()
        self.activation = nn.Sigmoid() if last else nn.LeakyReLU(0.2, inplace=True)

            
    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.activation(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.block1 = Block(1, 64, 4, 2, 1)
        self.block2 = Block(64, 128, 4, 2, 1)
        self.block3 = Block(128, 256, 4, 2, 1)
        self.block4 = Block(256, 512, 4, 2, 1)
        self.block5 = Block(512, 1, 4, 1, 0, norm=False, last=True)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        return x

