from torch.optim import AdamW
from pathlib import Path
import argparse

from utils import (
    get_device,
    get_grad_scaler,
    image_to_grid,
    save_image,
    merge_images_h,
)
from model import MAE
from imagenet1k import get_imagenet1k_train_dl


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)

    args = parser.parse_args()

    args_dict = vars(args)
    new_args_dict = dict()
    for k, v in args_dict.items():
        new_args_dict[k.upper()] = v
    args = argparse.Namespace(**new_args_dict)
    return args


def main():
    args = get_args()
    DEVICE = get_device()

    BATCH_SIZE = 4
    dl = get_imagenet1k_train_dl(args.DATA_DIR, batch_size=BATCH_SIZE, n_cpus=2)
    di = iter(dl)

    patch_size = 16
    img_size = 256
    enc_depth = 12
    enc_width = 768
    enc_n_heads = 12
    dec_depth = 2
    dec_width = 128
    dec_n_heads = 2
    model = MAE(
        img_size,
        patch_size,
        enc_depth,
        enc_width,
        enc_n_heads,
        dec_depth,
        dec_width,
        dec_n_heads,
    ).to(DEVICE)

    LR = 1e-3
    optim = AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.05)
    scaler = get_grad_scaler(device=DEVICE)

    N_EPOCHS = 600
    SAMPLES_DIR = Path(__file__).resolve().parent/"samples"
    for batch_idx in range(1, 11):
        ori_image, _ = next(di)
        ori_image = ori_image.to(DEVICE)

        ori_grid = image_to_grid(ori_image, n_cols=int(BATCH_SIZE ** 0.5))

        for epoch in range(1, N_EPOCHS + 1):
            loss = model.get_loss(ori_image)
            if epoch % 100 == 0 or epoch == N_EPOCHS:
                print(f"[ {epoch}/{N_EPOCHS} ][ {loss:.3f} ]")
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

        masked_image, recon_image = model.reconstruct(ori_image)
        masked_gird = image_to_grid(masked_image, n_cols=int(BATCH_SIZE ** 0.5))
        recon_grid = image_to_grid(recon_image, n_cols=int(BATCH_SIZE ** 0.5))
        
        merged_grid = merge_images_h([masked_gird, recon_grid, ori_grid])
        save_image(merged_grid, SAMPLES_DIR/f"{batch_idx}.jpg")


if __name__ == "__main__":
    main()
