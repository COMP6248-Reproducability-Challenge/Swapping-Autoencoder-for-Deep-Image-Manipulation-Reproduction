import math
import torch
from torch import nn
from stylegan2_pytorch.stylegan2_model import ConvLayer, ResBlock, EqualLinear


class PatchDiscriminator(nn.Module):
    """
        Co-occurrence patch discriminator -- determines if a patch in question (“real/fake patch”) is from the same
        image as a set of reference patches

        Takes a random sized patch (1/8 to 1/4 of the full image side-dimension up to 256) and outputs a scalar value
    """
    scale_up_to = 1024  # Patch size (dim*channels) to scale up to
    mc = 256 + 128  # Encoder output (maximum) number of channels to downscale to

    def __init__(self, patch_dim=128, channel_scale_factor=4.0):
        super().__init__()

        # Since the patches have random sizes, they are upscaled to the same size before being passed to the
        # co-occurrence discriminator
        num_down_samples = int(math.ceil(math.log(patch_dim, 2)))
        scale_channel_amount = int(self.scale_up_to / (2 ** num_down_samples) * channel_scale_factor)

        current_dim = patch_dim

        downscalers = []
        in_channels = scale_channel_amount
        while current_dim > 4:
            out_channels = min(in_channels * 2, self.mc)
            current_dim = current_dim / 2
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
    def forward(self, input_patch, reference_patches=None):
        encoded_input = self.patchEncoder(input_patch)

        if reference_patches is not None:
            # Average encodings over reference_patches (possibly in batches)
            # Not 100% sure this is correct but when it's integrated with real data that should let us know
            b, n, *_ = reference_patches.size()
            if reference_patches.ndim == 5:  # Batch of b sets of n patches (from b images)
                flattened_patches = reference_patches.flatten(0, 1)
            else:
                flattened_patches = reference_patches
            encoded_references = self.patchEncoder(flattened_patches)
            _, c, h, w = encoded_references.size()
            encoded_references = encoded_references.view(b, n, c, h, w)
            mean_reference_encodings = encoded_references.mean(1)
            combined_encoding = torch.cat((encoded_input.flatten(1), mean_reference_encodings.flatten(1)), 1)
        else:
            # Just for checking layer shapes not actual functionality
            combined_encoding = torch.cat((encoded_input, encoded_input), 1)

        combined_encoding = torch.flatten(combined_encoding, 1)
        pred = self.predictor(combined_encoding)
        return pred
