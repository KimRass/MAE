import torch
from torch.optim import AdamW

from utils import get_device, get_grad_scaler, image_to_grid
from model import MAE
from imagenet1k import get_imagenet1k_train_dl


if __name__ == "__main__":
    DEVICE = get_device()

    ROOT = "/home/jbkim/Documents/datasets/imagenet1k/train"
    BATCH_SIZE = 4
    dl = get_imagenet1k_train_dl(ROOT, batch_size=BATCH_SIZE, n_cpus=2)
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

    image, _ = next(di)
    image = image.to(DEVICE)

    N_EPOCHS = 800
    image_to_grid(image, n_cols=int(BATCH_SIZE ** 0.5)).show()
    for epoch in range(1, N_EPOCHS + 1):
        loss = model.get_loss(image)
        if epoch % 100 == 0 or epoch == N_EPOCHS:
            print(f"[ {epoch}/{N_EPOCHS} ][ {loss:.3f} ]")
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

    recon_image = model.reconstruct(image)
    image_to_grid(recon_image, n_cols=int(BATCH_SIZE ** 0.5)).show()
