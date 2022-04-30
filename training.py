import logging

import torch
from torch import optim
from torch import save, load

from data_loading import load_church_data
from swapping_autoencoder import SwappingAutoencoder, ForwardMode as Mode
from taesung_data_loading import ConfigurableDataLoader


class MultiGPUWrapper:
    def __init__(self, model: SwappingAutoencoder):
        if torch.cuda.device_count() > 1:
            print("Initialising model for ", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel(model).to(device)
        else:
            print("Running on single device ", device)
            self.model = model.to(device)

    def get_autoencoder_params(self):
        return self.model.get_autoencoder_params()

    def get_discriminator_params(self):
        return self.model.get_discriminator_params()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        return self.model.load_state_dict(state_dict)


class AutoencoderOptimiser:
    def __init__(self, image_crop_size):
        self.model = MultiGPUWrapper(SwappingAutoencoder(image_crop_size))
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

        return {"patchD_loss"}


def train(iterations: int, data_loader: ConfigurableDataLoader, image_crop_size: int, load_state=False):
    if load_state:
        optimiser = load_train_state(image_crop_size)
    else:
        optimiser = AutoencoderOptimiser(image_crop_size)

    training_discriminator = False
    for i in range(iterations):
        real_minibatch = next(data_loader)["real_A"].to(device)
        if training_discriminator:
            optimiser.train_discriminator_step(real_minibatch)
        else:
            optimiser.train_generator_step(real_minibatch)
        print("Iter:", i, "complete. Trained discriminator:", training_discriminator),
        training_discriminator = not training_discriminator

        # if i % 480 == 0:
        # TODO print current losses/metrics
        if i % 100 == 0:
            save_train_state(optimiser, i)
        # TODO evaluate metrics of model


def save_train_state(optimiser: AutoencoderOptimiser, i: int):
    save(optimiser.model.state_dict(), './saves/optimiser.pt')
    f = open("./saves/last_completed_iter.txt", "a")
    f.write(str(i))


def load_train_state(crop_size: int):
    new = AutoencoderOptimiser(crop_size)
    try:
        state_dict = load('./saves/optimiser.pt')
    except Exception:
        logging.exception("An error occurred whilst loading ./saves/optimiser.pt (has it been saved?)")
        return new

    try:
        new.model.load_state_dict(state_dict)
    except Exception:
        logging.exception("An error occurred whilst loading the state dictionary (is the model saved correctly?)")

    return new


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = "cuda"
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = "cpu"
    print("Running on ", device)
    torch.multiprocessing.set_start_method('spawn')
    print("Starting...")
    image_crop_size = 64
    print("Loading dataset")
    print("Loading data with ", torch.cuda.device_count(), "GPU workers")
    data_loader = load_church_data(image_crop_size=image_crop_size, batch_size=16, num_gpus=torch.cuda.device_count(),
                                   device=device)
    print("Dataset loaded")
    print("Starting training...")
    train(iterations=25 * 1000 ** 2, data_loader=data_loader, image_crop_size=image_crop_size)
