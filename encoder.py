import torch
from torch import nn
from stylegan2_pytorch.stylegan2_model import ResBlock, ConvLayer, EqualLinear


# Spatial Code refers to the Structure Code, and
# Global Code refers to the Texture Code of the paper.

class Encoder(torch.nn.Module):
    """
        Encoder - takes images and breaks down into two codes (Structure and Texture)
    """

    # TODO - confirm which params we want to take in and which to hardcode
    def __init__(self, no_downsamples=4, n_channels=32, structure_channels=8, antialias_used=False):
        super().__init__()

        # TODO is it necessary to do the blur kernel bit?
        if antialias_used:
            blur_kernel = [1, 2, 1]
        else:
            blur_kernel = [1]

        self.fromRGB = ConvLayer(3, n_channels, 1)

        # Creating the Downsampling Res-Blocks in loop
        resblocks = []
        for i in range(1, no_downsamples + 1):
            out_channels = n_channels * (2 ** 1)
            resblocks.append(ResBlock(n_channels, out_channels))
            n_channels = out_channels

        # Adding them sequentially
        self.DownSample = nn.Sequential(*resblocks)

        # TODO check parameters for structure and texture branches.
        #   Not sure if they are detaled in the paper, but I'm copying the implementation of their code
        # Structure branch
        self.structure = nn.Sequential(
            ConvLayer(n_channels, n_channels, 1),
            ConvLayer(n_channels, structure_channels, 1, activate=False)
            # TODO activate=False in their implementation,
            #   but not sure if it says in paper
        )

        # Texture Branch
        self.texture = nn.Sequential(
            ConvLayer(n_channels, n_channels*2, 3, downsample=True, blur_kernel=[1]),
            ConvLayer(n_channels * 2, n_channels * 4, 3, downsample=True, blur_kernel=[1]),
            nn.AdaptiveAvgPool2d(1),
            nn.EqualLinear(n_channels * 4, n_channels * 4)
        )

    def forward(self, x):
        out = self.fromRGB(x)
        out = self.DownSample(out)

        structure = self.structure(out)
        texture = torch.flatten(self.texture(out), 1)
        # TODO - In the code they take the mean(dim(2,3)) of the texture - I'm not sure why?

        structure = normalise(structure)
        texture = normalise(texture)
        return structure, texture


# Taken directly from their util.py
def normalise(v):
    if type(v) == list:
        return [normalise(vv) for vv in v]
    return v * torch.rsqrt((torch.sum(v ** 2, dim=1, keepdim=True) + 1e-8))
