from torch import optim
from torch.utils.data import DataLoader

from data_loading import load_church_data
from swapping_autoencoder import SwappingAutoencoder, ForwardMode as Mode


class AutoencoderOptimiser:
    def __init__(self, image_crop_size):
        self.model = SwappingAutoencoder(image_crop_size)
        self.autoencoder_params = self.model.get_autoencoder_params()
        self.optimiser_autoencoder = optim.Adam(self.autoencoder_params, lr=0.002, betas=(0.0, 0.99))
        self.r1_every = 16
        self.discriminator_params = self.model.get_discriminator_params()
        self.optimiser_discriminator = optim.Adam(self.discriminator_params,
                                                  lr=0.002 * self.r1_every / (1 + self.r1_every),
                                                  betas=(0.0, 0.99))
        self.discriminator_iterations = 0

    @staticmethod
    def toggle_params_grad(params, value):
        for p in params:
            p.requires_grad_(value)

    def train_generator_step(self, real_minibatch):
        self.toggle_params_grad(self.autoencoder_params, True)
        self.toggle_params_grad(self.discriminator_params, False)
        self.optimiser_autoencoder.zero_grad()
        loss = self.model(real_minibatch, Mode.AUTOENCODER_LOSS)
        loss.backward()
        self.optimiser_autoencoder.step()

    def train_discriminator_step(self, real_minibatch):
        self.toggle_params_grad(self.autoencoder_params, False)
        self.toggle_params_grad(self.discriminator_params, True)
        self.optimiser_discriminator.zero_grad()
        loss = self.model(real_minibatch, Mode.PATCH_DISCRIMINATOR_LOSS)
        loss.backward()
        self.optimiser_discriminator.step()

        # Calculate lazy-R1 regularisation
        if self.discriminator_iterations % self.r1_every == 0:
            self.optimiser_discriminator.zero_grad()
            r1_loss = self.model(real_minibatch, Mode.R1_LOSS)
            r1_loss *= self.r1_every
            r1_loss.backward()
            self.optimiser_discriminator.step()

        self.discriminator_iterations += 1


def train(iterations: int, data_loader: DataLoader, image_crop_size: int):
    optimiser = AutoencoderOptimiser(image_crop_size)

    training_discriminator = False
    for i in range(iterations):
        real_minibatch = next(iter(data_loader))["real_A"]
        if training_discriminator:
            optimiser.train_discriminator_step(real_minibatch)
        else:
            optimiser.train_generator_step(real_minibatch)
        training_discriminator = not training_discriminator

        # if i % 480 == 0:
        # TODO print current losses/metrics
        # if i % 50000 == 0:
        # TODO save model state and allow for re-loading of saved state (and number of iterations)
        # TODO evaluate metrics of model

def save_train_state():
    raise NotImplementedError()

if __name__ == '__main__':
    print("Starting...")
    image_crop_size = 256
    print("Loading dataset")
    data_loader = load_church_data(image_crop_size=image_crop_size, batch_size=16, num_gpus=0)
    print("Dataset loaded")
    print("Starting training...")
    train(iterations=25 * 1000 ** 2, data_loader=data_loader, image_crop_size=image_crop_size)
