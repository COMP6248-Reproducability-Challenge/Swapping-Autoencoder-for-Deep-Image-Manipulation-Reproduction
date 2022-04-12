from enum import Enum

import torch
from torch import nn

from decoder import Decoder
from encoder import Encoder
from patch_discriminator import PatchDiscriminator, get_random_patches
from stylegan2_pytorch.stylegan2_model import Discriminator as RealismDiscriminator


class ForwardMode(Enum):
    AUTOENCODER_LOSS = 0
    PATCH_DISCRIMINATOR_LOSS = 1
    R1_LOSS = 2


class SwappingAutoencoder(nn.Module):
    """
        Swapping autoencoder
    """

    def __init__(self, image_crop_size):
        super().__init__()
        self.encoder = Encoder()
        self.generator = Decoder()
        self.discriminator = RealismDiscriminator(image_crop_size)
        self.patch_discriminator = PatchDiscriminator()

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
        L_GAN_rec = -torch.log(self.discriminator(reconstructed)).view(N // 2, -1).mean(dim=1)
        L_GAN_swap = -torch.log(self.discriminator(swapped)).view(N // 2, -1).mean(dim=1)
        L_co_occur_GAN = -torch.log(self.patch_discriminator(get_random_patches(real_minibatch),
                                                             get_random_patches(swapped))).view(N, -1).mean(dim=1)
        # (?) author's code uses 1.0 * L_GAN_swap but I'm pretty sure that's an error as it doesn't match the paper
        L_GAN = 0.5 * L_GAN_rec + 0.5 * L_GAN_swap
        return L_rec + L_GAN + L_co_occur_GAN

    def calculate_patch_discriminator_loss(self, real_minibatch):
        """
            Standard GAN Minimax/Non-Saturating loss (Binary Cross-Entropy)
        """
        N = real_minibatch.size(0)  # batch_size
        reconstructed, swapped = self.generate_reconstructed_and_swapped(real_minibatch)
        # patch_disc estimate of whether the real image patches co-occur with themselves
        co_occurrence_real = self.patch_discriminator(get_random_patches(real_minibatch),
                                                      get_random_patches(real_minibatch))
        # patch_disc estimate of whether the fake image patches co-occur with the real ones
        co_occurrence_swapped = self.patch_discriminator(get_random_patches(swapped),
                                                         get_random_patches(real_minibatch))
        L_patch_real = -torch.log(co_occurrence_real).view(N, -1).mean(dim=1)
        L_patch_swapped = -torch.log(1 - co_occurrence_swapped).view(N, -1).mean(dim=1)

        L_GAN_real = -torch.log(self.discriminator(reconstructed)).view(N, -1).mean(dim=1)
        L_GAN_rec = -torch.log(1 - self.discriminator(reconstructed)).view(N // 2, -1).mean(dim=1)
        L_GAN_swap = -torch.log(1 - self.discriminator(swapped)).view(N // 2, -1).mean(dim=1)
        L_GAN_fake = 0.5 * L_GAN_rec + 0.5 * L_GAN_swap

        return L_patch_real + L_patch_swapped + L_GAN_real + L_GAN_fake

    def calculate_r1_loss(self, real_minibatch):
        """
            Loss for R1 regularisation of patch discriminator

            R1 term function from https://arxiv.org/pdf/1801.04406.pdf Section 4.1 Eq. 9
        """
        lambda_discriminator = 10.0
        lambda_patch = 1.0

        real_minibatch.requires_grad_(True)
        # d_discriminator/d_real
        grad_disc_real, = torch.autograd.grad(
            outputs=self.discriminator(real_minibatch).sum(),
            inputs=[real_minibatch]
        )
        grad_disc_square = grad_disc_real.pow(2)
        # Not 100% on this mean calculation, the author's code uses a summation over a list of dimensions
        R1_discriminator = lambda_discriminator / 2 * grad_disc_square.mean()

        real_patches = get_random_patches(real_minibatch)
        real_patches.requires_grad_(True)
        target_patches = get_random_patches(real_minibatch)
        target_patches.requires_grad_(True)
        # d_patch/d_real_patches and d_patch/d_target_patches
        grad_patch_real, grad_patch_target = torch.autograd.grad(
            outputs=self.patch_discriminator(target_patches, real_patches).sum(),
            inputs=[real_patches, target_patches]
        )
        grad_patch_square = 0.5 * grad_patch_real.pow(2) + 0.5 * grad_patch_target.pow(2)
        # Not 100% on this mean calculation, the author's code uses a summation over a list of dimensions
        R1_patch = lambda_patch / 2 * grad_patch_square.mean()

        return R1_discriminator + R1_patch

    def get_discriminator_params(self):
        return list(self.patch_discriminator.parameters()) + list(self.discriminator.parameters())

    def get_autoencoder_params(self):
        return list(self.generator.parameters()) + list(self.encoder.parameters())

    def forward(self, real_minibatch, mode: ForwardMode):
        if mode == ForwardMode.AUTOENCODER_LOSS:
            return self.calculate_autoencoder_loss(real_minibatch)
        elif mode == ForwardMode.PATCH_DISCRIMINATOR_LOSS:
            return self.calculate_patch_discriminator_loss(real_minibatch)
        elif mode == ForwardMode.R1_LOSS:
            return self.calculate_r1_loss(real_minibatch)
        else:
            raise ValueError(f"Unknown forward mode: {mode}")
