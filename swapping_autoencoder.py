from enum import Enum

import torch
from torch import nn

from patch_discriminator import PatchDiscriminator, get_random_patches

from stylegan2_pytorch.stylegan2_model import Discriminator as RealismDiscriminator

from training import image_crop_size


class ForwardMode(Enum):
    AUTOENCODER_LOSS = 0
    PATCH_DISCRIMINATOR_LOSS = 1
    R1_LOSS = 2


class SwappingAutoencoder(nn.Module):
    """
        Swapping autoencoder
    """

    def __init__(self):
        super().__init__()
        # self.encoder = Encoder()
        # self.generator = Generator()
        self.discriminator = RealismDiscriminator(image_crop_size)
        self.patchDiscriminator = PatchDiscriminator()

    def swap(self, images):
        """
            :param images: minibatch of images to swap
            :return: the minibatch with each subsequent pair of images swapped in position
                     E.g. [0, 1, 2, 3, 4, 5] -> [1, 0, 3, 2, 5, 4]
        """
        N, *image_shape = images.shape  # batch_size, image_shape[]
        assert N % 2 == 0
        new_shape = [N // 2, 2, *image_shape]
        images = images.view(*new_shape)
        images = torch.flip(images, [1])
        return images.view(N, *image_shape)

    def generate_reconstructed_and_swapped(self, real_minibatch):
        N = real_minibatch.size(0)  # batch_size
        structure, texture = self.encoder(real_minibatch)
        reconstructed = self.generator(structure[: N // 2], texture[: N // 2])
        structure_swapped = self.swap(structure)
        swapped = self.generator(structure_swapped, texture)
        return reconstructed, swapped

    def calculate_autoencoder_loss(self, real_minibatch):
        N = real_minibatch.size(0)  # batch_size
        reconstructed, swapped = self.generate_reconstructed_and_swapped(real_minibatch)
        L1 = nn.L1Loss(reduction='mean')
        L_rec = L1(reconstructed, real_minibatch[: N // 2])
        # (?) code from paper uses F.softplus(-D) where softplus(x) = log(1 + e^x)
        #   I think this \/ the Non-saturating GAN loss (which they say they use) and this /\ is the minimax GAN loss?
        L_GAN_rec = -torch.log(self.discriminator(reconstructed)).view(N, -1).mean(dim=1)
        L_GAN_swap = -torch.log(self.discriminator(swapped)).view(N, -1).mean(dim=1)
        L_co_occur_GAN = -torch.log(self.patchDiscriminator(get_random_patches(real_minibatch),
                                                            get_random_patches(swapped))).view(N, -1).mean(dim=1)
        # (?) code from paper uses 1.0 * L_GAN_swap
        return L_rec + 0.5 * L_GAN_rec + 0.5 * L_GAN_swap + L_co_occur_GAN

    def calculate_patch_discriminator_loss(self, real_minibatch):
        """
            Standard GAN Minimax/Non-Saturating loss (Binary Cross-Entropy)
        """
        N = real_minibatch.size(0)  # batch_size
        reconstructed, swapped = self.generate_reconstructed_and_swapped(real_minibatch)
        # patch_disc estimate of whether the real image patches co-occur with themselves
        co_occurrence_real = self.patchDiscriminator(get_random_patches(real_minibatch),
                                                     get_random_patches(real_minibatch))
        # patch_disc estimate of whether the fake image patches co-occur with the real ones
        co_occurrence_swapped = self.patchDiscriminator(get_random_patches(swapped),
                                                        get_random_patches(real_minibatch))
        L_real = -torch.log(co_occurrence_real).view(N, -1).mean(dim=1)
        L_swapped = -torch.log(1 - co_occurrence_swapped).view(N, -1).mean(dim=1)
        # TODO I think they also add in the image (GAN) discriminator losses as well?
        return L_real + L_swapped

    def calculate_r1_loss(self, real_minibatch):
        """
            Loss for R1 regularisation of patch discriminator
        """
        # TODO see StyleGAN2 (?)
        raise NotImplementedError()

    def get_discriminator_params(self):
        return self.patchDiscriminator.parameters()

    def get_autoencoder_params(self):
        raise list(self.generator.parameters()) + list(self.encoder.parameters())

    def forward(self, real_minibatch, mode: ForwardMode):
        if mode == ForwardMode.AUTOENCODER_LOSS:
            return self.calculate_autoencoder_loss(real_minibatch)
        elif mode == ForwardMode.PATCH_DISCRIMINATOR_LOSS:
            return self.calculate_patch_discriminator_loss(real_minibatch)
        elif mode == ForwardMode.R1_LOSS:
            return self.calculate_r1_loss(real_minibatch)
        else:
            raise ValueError(f"Unknown forward mode: {mode}")
