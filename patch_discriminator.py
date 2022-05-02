import math

import torch
from torch import nn

from stylegan2_pytorch.stylegan2_model import ConvLayer, ResBlock, EqualLinear
from taesung_data_loading.util import apply_random_crop


def get_random_patches(images, num_crops=8, min_scale=1 / 8, max_scale=1 / 4):
    """
        Generate num_crops random patches from each image in images
    """
    if images.ndim == 4:  # Batch of b images
        b, c, h, w = images.shape
    elif images.ndim == 3:  # Single image
        c, h, w = images.shape
    else:
        raise NotImplementedError("get_random_patches not implemented for tensor of dimensionality ", images.ndim,
                                  ". Shape: ", images.shape)
    patch_dim = h // 4
    return apply_random_crop(images, patch_dim, (min_scale, max_scale), num_crops)


class PatchDiscriminator(nn.Module):
    """
        Co-occurrence patch discriminator -- determines if a patch in question (“real/fake patch”) is from the same
        image as a set of reference patches

        Takes a random sized patch (1/8 to 1/4 of the full image side-dimension up to 256) and outputs a scalar value

        All patches should be scaled to the same size (patch_dim) (1/4 of image side-dimension) before being passed to
        the discriminator -- Might need to be a power of 2, default: patch_dim=128
    """
    scale_up_to = 1024  # Patch size (dim*channels) to scale up to
    mc = 256 + 128  # Encoder output (maximum) number of channels to downscale to

    def __init__(self, patch_dim=128, channel_scale_factor=4.0):
        super().__init__()

        # Convolve 3 channels into a power of 2
        num_down_samples = int(math.ceil(math.log(patch_dim, 2)))
        scale_channel_amount = int(self.scale_up_to / (2 ** num_down_samples) * channel_scale_factor)

        downscalers = []
        in_channels = scale_channel_amount
        for sample in range(num_down_samples, 2, -1):
            out_channels = min(in_channels * 2, self.mc)
            # Assuming antialiasing blur_filter
            downscalers.append(ResBlock(in_channels, out_channels))
            in_channels = out_channels

        # Each patch is first independently encoded
        self.patchEncoder = nn.Sequential(
            # Upscale to constant patch size
            ConvLayer(3, scale_channel_amount, 3),
            # Downscale to max_channels (4x4) (halve layer size and double channels)
            *downscalers,
            ResBlock(self.mc, self.mc * 2, downsample=False),
            ConvLayer(self.mc * 2, self.mc, 3, pad=0)
        )

        encoder_output_size = self.mc * 2 * 2  # max-channels * 2x2 feature image

        # Then 3 dense layers are applied to classify the final prediction
        self.predictor = nn.Sequential(
            EqualLinear(encoder_output_size * 2, 2048, activation='fused_lrelu'),  # For 2 images concatenated
            EqualLinear(2048, 2048, activation='fused_lrelu'),
            EqualLinear(2048, 1024, activation='fused_lrelu'),
            EqualLinear(1024, 1)
        )

    # input_patch = fake generated patch
    def forward(self, input_patch, reference_patches):
        # Average encodings over reference_patches (possibly in batches)
        b, n, *_ = input_patch.size()
        if input_patch.ndim == 5:  # Batch of b sets of n patches (from b images)
            flattened_patches = input_patch.flatten(0, 1)
        else:
            flattened_patches = input_patch
        encoded_inputs = self.patchEncoder(flattened_patches)
        _, c, h, w = encoded_inputs.size()
        encoded_inputs = encoded_inputs.view(b, n, c, h, w)
        mean_input_encodings = encoded_inputs.mean(1)

        # Average encodings over reference_patches (possibly in batches)
        b, n, *_ = reference_patches.size()
        if reference_patches.ndim == 5:  # Batch of b sets of n patches (from b images)
            flattened_patches = reference_patches.flatten(0, 1)
        else:
            flattened_patches = reference_patches
        encoded_references = self.patchEncoder(flattened_patches)
        _, c, h, w = encoded_references.size()
        encoded_references = encoded_references.view(b, n, c, h, w)
        mean_reference_encodings = encoded_references.mean(1)

        combined_encoding = torch.cat((mean_input_encodings.flatten(1), mean_reference_encodings.flatten(1)), 1)
        combined_encoding = torch.flatten(combined_encoding, 1)
        pred = self.predictor(combined_encoding)
        return pred
