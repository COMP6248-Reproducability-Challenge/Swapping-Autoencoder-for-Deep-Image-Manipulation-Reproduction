import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2

from training import *

save_path = '../cwk_checkpoints/56750/optimiser.pt'


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
    image_crop_size = 64
    batch_size = 128
    data_loader = load_church_data(image_crop_size=image_crop_size, batch_size=batch_size, num_gpus=0, device=device)
    print("Dataset loaded")
    print("Loading model")
    optimiser = AutoencoderOptimiser(image_crop_size)
    state_dict = load(save_path)
    optimiser.model.load_state_dict(state_dict)
    print("Model loaded")

    plt.figure(figsize=(8, 4))
    model = optimiser.model.model
    batch = next(data_loader)["real_A"]
    for i in range(4):
        # Display original
        img = batch[np.random.randint(0, batch_size)]
        plt.subplot(int(str(24) + str(2 * i + 1)))
        plt.imshow(format_as_image(img))
        plt.axis('off')

        # Calculate and display reconstruction
        plt.subplot(int(str(24) + str(2 * i + 2)))
        structure, texture = model.encoder(tensor_rgb_to_bgr(img).unsqueeze(0))
        model_data = model.generator(structure, texture).squeeze().detach()
        plt.imshow(bgr_to_rgb(format_as_image(model_data).numpy()))
        plt.axis('off')

    plt.show()

    # Number of images in the grid
    grid_size = 5
    texture_imgs = [batch[np.random.randint(0, batch_size)] for _ in range(grid_size)]
    structure_imgs = [batch[np.random.randint(0, batch_size)] for _ in range(grid_size)]

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
            _, texture = model.encoder(tensor_rgb_to_bgr(img_t).unsqueeze(0))
            structure, _ = model.encoder(tensor_rgb_to_bgr(img_s).unsqueeze(0))
            model_data = model.generator(structure, texture).squeeze().detach()
            plt.imshow(bgr_to_rgb(format_as_image(model_data).numpy()))
            plt.axis('off')
    plt.show()
    print()
