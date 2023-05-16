import torch
import numpy as np


def get_img(img_tensor: torch.Tensor):
    """Gets a data point as image.

    Args:
        img_tensor (torch.Tensor): The image of shape CxHxW in the range [0-1]
    """
    img = 255.0 * torch.movedim(img_tensor, 0, 2).cpu().detach().numpy()
    if img.shape[-1] == 1:
        img = np.repeat(img, repeats=3, axis=-1)
    img = img.astype("uint8")
    return img
