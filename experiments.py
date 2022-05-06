import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2

from training import *

save_path = '../cwk_checkpoints/run-2/final/optimiser.pt'
invert_channels = False


#  Convert tensor image data (3, N, M) to np image data format (N, M, 3) and range 0--1
def format_as_image(data: torch.Tensor):
    return (data.permute((1, 2, 0)) + 1) / 2


#  Convert image data (N, M, 3) from BGR to RGB
def bgr_to_rgb(data: np.ndarray):
    return cv2.cvtColor(data, cv2.COLOR_BGR2RGB)


#  Convert tensor image data (3, N, M) from RGB to BGR
def tensor_rgb_to_bgr(data: torch.Tensor):
    np_data = format_as_image(data).detach().cpu().numpy()
    corrected_np_data = cv2.cvtColor(np_data, cv2.COLOR_RGB2BGR)
    return torch.Tensor(corrected_np_data).to(device).permute((2, 0, 1)) * 2 - 1


if __name__ == '__main__':
    device = "cpu"
    print("Loading dataset")
    batch_size = 1024
    image_crop_size = 64
    data_loader = load_church_data(image_crop_size=image_crop_size, batch_size=batch_size, num_gpus=0, device=device,
                                   dir_path="../CwkData/lsun/church_outdoor_train_lmdb")
    print("Dataset loaded")
    print("Loading model")
    optimiser = AutoencoderOptimiser(64)
    state_dict = load(save_path)
    optimiser.model.load_state_dict(state_dict)
    print("Model loaded")

    # Day -> Night modification figures -- need to be first as they use the first data batch
    model = optimiser.model.model
    batch = next(data_loader)["real_A"]
    day = [4, 260, 196, 908, 857, 869, 551, 555, 557, 447]
    day_imgs = [batch[idx] for idx in day]
    night = [12, 797, 324, 147, 815, 939, 910, 960, 962, 975]
    night_imgs = [batch[idx] for idx in night]
    _, day_textures = model.encoder(torch.cat([img.unsqueeze(0) for img in day_imgs]))
    _, night_textures = model.encoder(torch.cat([img.unsqueeze(0) for img in night_imgs]))
    day_avg = day_textures.mean(dim=0)
    night_avg = night_textures.mean(dim=0)
    diff = night_avg - day_avg
    for n, idx in enumerate([27, 26, 28, 23]):
        inpt = batch[idx]
        structure, texture = model.encoder(inpt.unsqueeze(0))
        plt.figure(figsize=(9, 1.3))
        for i, gain in enumerate(np.linspace(-1.5, 2.5, 9)):
            model_data = model.generator(structure, texture + gain * diff).squeeze().detach()
            plt.subplot(1, 9, i + 1)
            if gain != 0:
                plt.imshow(format_as_image(model_data).numpy())
            else:
                plt.imshow(format_as_image(inpt).numpy())
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"../CwkResults/day-night-{n}.jpg")

    for n in range(50):
        plt.figure(figsize=(8, 4))
        for i in range(4):
            # Display original
            img = data_loader.underlying_dataset[np.random.randint(0, len(data_loader.underlying_dataset))]['real_A']
            plt.subplot(int(str(24) + str(2 * i + 1)))
            plt.imshow(format_as_image(img))
            plt.axis('off')

            # Calculate and display reconstruction
            plt.subplot(int(str(24) + str(2 * i + 2)))
            structure, texture = model.encoder((tensor_rgb_to_bgr(img) if invert_channels else img).unsqueeze(0))
            model_data = model.generator(structure, texture).squeeze().detach()
            out_data = bgr_to_rgb(format_as_image(model_data).numpy()) if invert_channels else format_as_image(
                model_data).numpy()
            plt.imshow(out_data)
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(f"../CwkResults/recon{n}.jpg")

        # Number of images in the grid
        grid_size = 5
        texture_imgs = [
            data_loader.underlying_dataset[np.random.randint(0, len(data_loader.underlying_dataset))]['real_A']
            for _ in range(grid_size)]
        structure_imgs = [
            data_loader.underlying_dataset[np.random.randint(0, len(data_loader.underlying_dataset))]['real_A']
            for _ in range(grid_size)]

        plt.figure(figsize=((grid_size + 1) * 2, (grid_size + 1) * 2))
        # Display texture images along top
        for i, img in enumerate(texture_imgs):
            plt.subplot(grid_size + 1, grid_size + 1, i + 2)
            plt.imshow(format_as_image(img))
            plt.axis('off')

        # Display structure images on left
        for i, img in enumerate(structure_imgs):
            plt.subplot(grid_size + 1, grid_size + 1, (i + 1) * (grid_size + 1) + 1)
            plt.imshow(format_as_image(img))
            plt.axis('off')

        # Calculate and display image swaps
        for j, img_s in enumerate(structure_imgs):
            for i, img_t in enumerate(texture_imgs):
                subplot_i = (j + 1) * (grid_size + 1) + 2 + i
                plt.subplot(grid_size + 1, grid_size + 1, subplot_i)
                _, texture = model.encoder((tensor_rgb_to_bgr(img_t) if invert_channels else img_t).unsqueeze(0))
                structure, _ = model.encoder((tensor_rgb_to_bgr(img_s) if invert_channels else img_s).unsqueeze(0))
                model_data = model.generator(structure, texture).squeeze().detach()
                out_data = bgr_to_rgb(format_as_image(model_data).numpy()) if invert_channels else format_as_image(
                    model_data).numpy()
                plt.imshow(out_data)
                plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"../CwkResults/swap{n}.jpg")
    print()
