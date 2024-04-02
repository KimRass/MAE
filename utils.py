import torch
import torchvision.transforms as T
from torch.cuda.amp import GradScaler
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from time import time
from datetime import timedelta
from einops import rearrange
import numpy as np
import random
import os
import cv2


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
