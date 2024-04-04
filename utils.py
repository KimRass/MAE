import torch
from torch.cuda.amp import GradScaler
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from pathlib import Path
import numpy as np
import random
import os
import math
from PIL import Image


def denorm(x, mean=(0.457, 0.437, 0.404), std=(0.275, 0.271, 0.284)):
    return TF.normalize(
        x, mean=-(np.array(mean) / np.array(std)), std=(1 / np.array(std)),
    )


@torch.inference_mode()
def image_to_grid(image, n_cols, padding=1, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    tensor = image.clone().detach().cpu()
    tensor = denorm(tensor, mean=mean, std=std)
    grid = make_grid(tensor, nrow=n_cols, padding=padding, pad_value=1)
    grid.clamp_(0, 1)
    grid = TF.to_pil_image(grid)
    return grid


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    return device


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    return device


def get_grad_scaler(device):
    return GradScaler() if device.type == "cuda" else None


def create_dir(x):
    x = Path(x)
    if x.suffix:
        x.parent.mkdir(parents=True, exist_ok=True)
    else:
        x.mkdir(parents=True, exist_ok=True)


def to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def save_image(image, save_path):
    create_dir(save_path)
    to_pil(image).save(str(save_path), quality=100)
    print(f"""Saved image as "{Path(save_path).name}".""")


def merge_images_h(images):
    sum_w = 0
    min_h = math.inf
    for image in images:
        w, h = image.size
        sum_w += w
        min_h = min(min_h, h)
    merged_image = Image.new(mode="RGB", size=(sum_w, min_h))

    sum_w = 0
    for image in images:
        merged_image.paste(image, (sum_w, 0))
        sum_w += image.width
    return merged_image
