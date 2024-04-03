# "Our MAE works well using cropping-only augmentation, either fixed-size or random-size (both having random horizontal flipping). Adding color jittering degrades the results and so we do not use it in other experiments."
# "This property is dramatically different from contrastive learning and related methods [62, 23, 7, 21], which heavily rely on data augmentation. It was observed [21] that using cropping-only augmentation reduces the accuracy by 13% and 28% respectively for BYOL [21] and SimCLR."
# "The role of data augmentation is mainly performed by random masking (ablated next). The masks are different for each iteration and so they generate new training samples regardless of data augmentation. The pretext task is made difficult by masking and requires less augmentation to regularize training."


from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2

from utils import image_to_grid


def get_imagenet1k_train_dl(root, batch_size, n_cpus, img_size=256):
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            # A.SmallestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            ),
            A.CenterCrop(height=img_size, width=img_size),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ],
    )
    ds = ImageFolder(
        root=root,
        transform=lambda x: transform(image=np.array(x))["image"],
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        num_workers=n_cpus,
    )


if __name__ == "__main__":
    root = "/home/jbkim/Documents/datasets/imagenet1k/train"
    dl = get_imagenet1k_train_dl(root, batch_size=4, n_cpus=2)
    di = iter(dl)
    image, label = next(di)
    image_to_grid(image, n_cols=2).show()
