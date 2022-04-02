import torch
from torch import optim
from torch.utils.data import DataLoader, IterableDataset


class AutoencoderOptimiser:
    def __init__(self):
        self.model = SwappingAutoencoder()
        self.discriminator_params = self.model.get_discriminator_params()
        self.autoencoder_params = self.model.get_autoencoder_params()
        self.optimiser_autoencoder = optim.Adam(self.discriminator_params, lr=0.002, betas=(0.0, 0.99))
        self.r1_every = 16
        self.optimiser_discriminator = optim.Adam(self.autoencoder_params,
                                                  lr=0.002 * self.r1_every / (1 + self.r1_every),
                                                  betas=(0.0, 0.99))
        self.discriminator_iterations = 0

    @staticmethod
    def toggle_params_grad(params, value):
        for p in params:
            p.requires_grad_(value)

    def train_generator_step(self, real_minibatch):
        # Switch on autoencoder gradient and switch off discriminator
        self.toggle_params_grad(self.autoencoder_params, True)
        self.toggle_params_grad(self.discriminator_params, False)
        self.optimiser_autoencoder.zero_grad()
        fake_minibatch = self.model(real_minibatch)
        loss = self.model.calculate_autoencoder_loss(real_minibatch, fake_minibatch)
        loss.backward()
        self.optimiser_autoencoder.step()

    def train_discriminator_step(self, real_minibatch):
        self.toggle_params_grad(self.autoencoder_params, False)
        self.toggle_params_grad(self.discriminator_params, True)
        self.optimiser_discriminator.zero_grad()
        fake_minibatch = self.model(real_minibatch)
        loss = self.model.calculate_discriminator_loss(real_minibatch, fake_minibatch)
        loss.backward()
        self.optimiser_discriminator.step()

        # Calculate lazy-R1 regularisation
        if self.discriminator_iterations % self.r1_every == 0:
            self.optimiser_discriminator.zero_grad()
            r1_loss = self.model.calculate_r1_losses(real_minibatch)
            r1_loss *= self.r1_every
            r1_loss.backward()
            self.optimiser_discriminator.step()

        self.discriminator_iterations += 1


def train(iterations: int, data_loader: DataLoader):
    optimiser = AutoencoderOptimiser()

    training_discriminator = False
    for _ in range(iterations):
        real_minibatch = next(iter(data_loader))
        if training_discriminator:
            optimiser.train_discriminator_step(real_minibatch)
        else:
            optimiser.train_generator_step(real_minibatch)
        training_discriminator = not training_discriminator
    raise NotImplementedError()


if __name__ == '__main__':
    dataset = load_data()
    loader = create_data_loader(dataset)
    train(iterations=25 * 1000 ** 2, data_loader=loader)
