import logging
import sys
from datetime import datetime

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
            self.parallel_model = torch.nn.DataParallel(model).to(device)
        else:
            print("Running on single device ", device)
            self.parallel_model = model.to(device)
        self.model = model.to(device)

    def get_autoencoder_params(self):
        return self.model.get_autoencoder_params()

    def get_discriminator_params(self):
        return self.model.get_discriminator_params()

    def __call__(self, *args, **kwargs):
        return self.parallel_model(*args, **kwargs)

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
        loss = self.model(real_minibatch, Mode.AUTOENCODER_LOSS).sum()
        loss.backward()
        self.optimiser_autoencoder.step()
        return {"autoE_loss": loss}

    def train_discriminator_step(self, real_minibatch):
        self.toggle_params_grad(self.autoencoder_params, False)
        self.toggle_params_grad(self.discriminator_params, True)
        self.optimiser_discriminator.zero_grad()
        loss = self.model(real_minibatch, Mode.PATCH_DISCRIMINATOR_LOSS).sum()
        loss.backward()
        self.optimiser_discriminator.step()

        ret_losses = {"pathD_loss": loss}
        # Calculate lazy-R1 regularisation
        if self.discriminator_iterations % self.r1_every == 0:
            self.optimiser_discriminator.zero_grad()
            r1_loss = self.model(real_minibatch, Mode.R1_LOSS).sum()
            r1_loss *= self.r1_every
            r1_loss.backward()
            self.optimiser_discriminator.step()
            ret_losses["disc_loss"] = r1_loss

        self.discriminator_iterations += 1

        return ret_losses


def train(iterations: int, data_loader: ConfigurableDataLoader, image_crop_size: int, start_i: int = 0,
          load_state=False):
    if load_state:
        optimiser = load_train_state(image_crop_size)
    else:
        optimiser = AutoencoderOptimiser(image_crop_size)

    print("Time:", datetime.now().strftime("%H:%M:%S"))
    training_discriminator = False
    for i in range(start_i, iterations):
        real_minibatch = next(data_loader)["real_A"].to(device)
        if training_discriminator:
            losses = optimiser.train_discriminator_step(real_minibatch)
        else:
            losses = optimiser.train_generator_step(real_minibatch)
        training_discriminator = not training_discriminator

        if i % 100 == 0:
            print(f"{i}/{iterations}. \t\tTime:", datetime.now().strftime("%H:%M:%S"), "\tLosses:", losses)
            save_train_state(optimiser, i)


def save_train_state(optimiser: AutoencoderOptimiser, i: int):
    save(optimiser.model.state_dict(), './saves/optimiser.pt')
    f = open("./saves/last_completed_iter.txt", "a")
    f.write(str(i) + "\n")
    f.close()


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
    argv = sys.argv
    if len(argv) != 2:
        print("Usage python training.py iter_start")
        sys.exit(2)
    _, start_i = argv
    start_i = int(start_i)
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
    batch_size = 128
    data_loader = load_church_data(image_crop_size=image_crop_size, batch_size=batch_size, num_gpus=0, device=device)
    print("Dataset loaded")
    print("Starting training from iteration ", start_i, "...")
    load_state = bool(start_i == 0)
    if load_state:
        print("Reloading training state from saves/optimiser.pt")
    else:
        print("Re-initialising model")
    train(start_i=start_i, iterations=int(25 * 1000 ** 2 // batch_size), data_loader=data_loader,
          image_crop_size=image_crop_size, load_state=load_state)
