import torch
from torch import nn

from stylegan2_pytorch.stylegan2_model import ResBlock, ConvLayer, EqualLinear


# Spatial Code refers to the Structure Code, and
# Global Code refers to the Texture Code of the paper.

class Encoder(torch.nn.Module):
    """
        Encoder - takes images and breaks down into two codes (Structure and Texture)

        small_images = True if last layer needs to be 1x1 convolution (e.g. crop_width <= 64)
    """

    def __init__(self, no_downsamples=4, n_channels=32, structure_channels=8, antialias_used=True, small_images=False):
        super().__init__()

        if antialias_used:
            blur_kernel = [1, 2, 1]
        else:
            blur_kernel = [1]

        self.fromRGB = ConvLayer(3, n_channels, 1)

        # Creating the Downsampling Res-Blocks in loop
        resblocks = []
        for i in range(no_downsamples):
            out_channels = n_channels * (2 ** 1)
            resblocks.append(ResBlock(n_channels, out_channels, blur_kernel, reflection_pad=True))
            n_channels = out_channels

        # Adding them sequentially
        self.DownSample = nn.Sequential(*resblocks)

        # Structure branch
        self.structure = nn.Sequential(
            ConvLayer(n_channels, n_channels, 1),
            ConvLayer(n_channels, structure_channels, 1, activate=False)
            #   activate=False removes the relu from the conv layer
        )

        # Texture Branch
        self.texture = nn.Sequential(
            ConvLayer(n_channels, n_channels * 2, 3, downsample=True, blur_kernel=[1], pad=0),
            ConvLayer(n_channels * 2, n_channels * 4, 1 if small_images else 3, downsample=True, blur_kernel=[1],
                      pad=0),
            nn.AdaptiveAvgPool2d(1),
            EqualLinear(n_channels * 4, n_channels * 4)
        )

    def forward(self, x):
        out = self.fromRGB(x)
        out = self.DownSample(out)

        structure = self.structure(out)
        texture = torch.flatten(self.texture(out), 1)

        structure = normalise(structure)
        texture = normalise(texture)
        return structure, texture


# Taken directly from their util.py
def normalise(v):
    if type(v) == list:
        return [normalise(vv) for vv in v]
    return v * torch.rsqrt((torch.sum(v ** 2, dim=1, keepdim=True) + 1e-8))
