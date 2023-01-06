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
                last: bool=False
        ):
        super(Block, self).__init__()

        self.conv = nn.ConvTranspose2d(channels_in, channels_out, kernel_size, stride, padding, bias=False)
        self.batchnorm = nn.BatchNorm2d(channels_out) if norm else nn.Identity()
        self.activation = nn.Sigmoid() if last else nn.ReLU(True)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.activation(x)

        return x


class Generator(nn.Module):
    def __init__(self, latent_vector_size: int):
        super(Generator, self).__init__()

        self.block1 = Block(latent_vector_size, 512, 4, 1, 0)
        self.block2 = Block(512, 256, 4, 2, 1)
        self.block3 = Block(256, 128, 4, 2, 1)
        self.block4 = Block(128, 64, 4, 2, 1)
        self.block5 = Block(64, 1, 4, 2, 1, norm=False, last=True)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        return x