# References:
    # https://github.com/KimRass/ViT/blob/main/model.py
    # https://github.com/KimRass/Transformer/blob/main/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import torchvision.transforms.functional as TF


class PatchEmbedding(nn.Module):
    """
    "our encoder embeds patches by a linear projection with added positional embeddings."
    """
    def __init__(self, patch_size, embed_dim, drop_prob=0.1):
        super().__init__()

        self.patch_size = patch_size

        in_feats = (patch_size ** 2) * 3
        self.layers = nn.Sequential(
            nn.LayerNorm(in_feats),
            nn.Linear(in_feats, embed_dim),
            nn.Dropout(drop_prob),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        x = einops.rearrange(
            x,
            pattern="b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        return self.layers(x)


class PositionalEmbedding(nn.Module):
    """
    "Our MAE adds positional embeddings (the sine-cosine version) to both the encoder
    and decoder inputs."
    """
    def __init__(self, embed_dim, max_len=2 ** 12):
        super().__init__()

        pos = torch.arange(max_len).unsqueeze(1)
        i = torch.arange(embed_dim // 2).unsqueeze(0)
        angle = pos / (10_000 ** (2 * i / embed_dim))

        self.pe_mat = torch.zeros(size=(max_len, embed_dim))
        self.pe_mat[:, 0:: 2] = torch.sin(angle)
        self.pe_mat[:, 1:: 2] = torch.cos(angle)
        self.register_buffer("pos_enc_mat", self.pe_mat)


class MSA(nn.Module):
    def __init__(self, width, n_heads, drop_prob=0.1):
        super().__init__()

        assert width % n_heads == 0, (
            "`width` must be divisible by `n_heads`!",
        )
        self.head_width = width // n_heads
        self.n_heads = n_heads

        self.qkv_proj = nn.Linear(width, 3 * n_heads * self.head_width, bias=False)
        self.scale = width ** (-0.5)
        self.drop = nn.Dropout(drop_prob)
        self.out_proj = nn.Linear(width, width, bias=False)

    def forward(self, x):
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=2)
        batch_size = x.size(0)
        attn_score = torch.einsum(
            "binh,bjnh->bijn",
            q.view(batch_size, -1, self.n_heads, self.head_width),
            k.view(batch_size, -1, self.n_heads, self.head_width)
        ) * self.scale
        attn_weight = F.softmax(attn_score, dim=2)
        x = torch.einsum(
            "bijn,bjnh->binh",
            attn_weight,
            k.view(batch_size, -1, self.n_heads, self.head_width)
        )
        x = einops.rearrange(x, pattern="b i n h -> b i (n h)")
        x = self.out_proj(x)
        return self.drop(x)
# class MSA(nn.Module):
#     def __init__(self, width, n_heads, drop_prob=0.1):
#         super().__init__()

#         assert width % n_heads == 0, (
#             "`width` must be divisible by `n_heads`!",
#         )
#         self.head_size = width // n_heads
#         self.n_heads = n_heads

#         self.qkv_proj = nn.Linear(width, 3 * n_heads * self.head_size, bias=False)
#         self.drop = nn.Dropout(drop_prob)
#         self.out_proj = nn.Linear(width, width, bias=False)

#     def _rearrange(self, x):
#         return einops.rearrange(
#             x, pattern="b n (h d) -> b h n d", h=self.n_heads, d=self.head_size,
#         )

#     @staticmethod
#     def _get_attention_score(q, k):
#         attn_score = torch.einsum("bhnd,bhmd->bhnm", q, k)
#         return attn_score

#     def forward(self, x):
#         qkv = self.qkv_proj(x)
#         q, k, v = torch.chunk(qkv, chunks=3, dim=2)
#         q = self._rearrange(q)
#         k = self._rearrange(k)
#         v = self._rearrange(v)
#         attn_score = self._get_attention_score(q=q, k=k)
#         attn_weight = F.softmax(attn_score / (self.head_size ** 0.5), dim=3)
#         x = torch.einsum("bhnm,bhmd->bhnd", attn_weight, v)
#         x = einops.rearrange(x, pattern="b h n d -> b n (h d)")
#         x = self.out_proj(x)
#         x = self.drop(x)
#         return x


class SkipConnection(nn.Module):
    def __init__(self, width):
        super().__init__()

        self.norm = nn.LayerNorm(width)

    def forward(self, x, sublayer):
        skip = x.clone()
        x = self.norm(x)
        x = sublayer(x)
        x += skip
        return x


class MLP(nn.Module):
    def __init__(self, width, drop_prob=0.1):
        super().__init__()

        self.mlp_width = width * 4

        self.layers = nn.Sequential(
            nn.Linear(width, self.mlp_width),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(self.mlp_width, width),
            nn.GELU(),
            nn.Dropout(drop_prob),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, width, n_heads):
        super().__init__()

        self.self_attn = MSA(width=width, n_heads=n_heads)
        self.self_attn_resid = SkipConnection(width=width)
        self.mlp = MLP(width=width)
        self.mlp_resid = SkipConnection(width=width)

    def forward(self, x):
        x = self.self_attn_resid(x=x, sublayer=self.self_attn)
        x = self.mlp_resid(x=x, sublayer=self.mlp)
        return x


class TransformerBlocks(nn.Module):
    """
    "We use xavier uniform to initialize all Transformer blocks."
    """
    def __init__(self, depth, width, n_heads):
        super().__init__()

        self.tf_layers = nn.ModuleList(
            [
                TransformerBlock(width=width, n_heads=n_heads)
                for _
                in range(depth)
            ]
        )

    def forward(self, x):
        for layer in self.tf_layers:
            x = layer(x)
        return x


class PatchShuffle(nn.Module):
    """
    "Next we randomly shuffle the list of tokens and remove the last portion of the list, based on the masking ratio. This process produces a small subset of tokens for the encoder and is equivalent to sampling patches without replacement."
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, mask_ratio):
        n_tokens = x.size(1)
        shuffled_indices = torch.randperm(n_tokens)
        mask_indices = shuffled_indices[-int(n_tokens * mask_ratio):]
        mask = torch.isin(torch.arange(n_tokens), mask_indices)
        return x[:, ~mask, :], mask


class Encoder(nn.Module):
    """
    "First we generate a token for every input patch (by linear projection with an added
    positional embedding)."
    "Our encoder is a ViT but applied only on visible, unmasked patches. Just as in a standard ViT, then processes the resulting set via a series of Transformer blocks. However, our encoder only operates on a small subset (e.g., 25%) of the full set. Masked patches are removed; no mask tokens are used."
    "After encoding, we append a list of mask tokens to the list of encoded patches, and
    unshuffle this full list (inverting the random shuffle operation) to align all tokens
    with their targets."
    "The encoder ends with LN."
    """
    def __init__(
        self,
        img_size,
        patch_size,
        depth,
        width,
        n_heads,
    ):
        super().__init__()

        self.width = width
        self.cell_size = img_size // patch_size

        self.patch_embed = PatchEmbedding(patch_size=patch_size, embed_dim=width)
        self.pos_embed = PositionalEmbedding(embed_dim=width)
        self.patch_shuffle = PatchShuffle()
        self.tf_block = TransformerBlocks(
            depth=depth,
            width=width,
            n_heads=n_heads,
        )
        self.norm = nn.LayerNorm(width)
        self.mask_token = nn.Parameter(torch.randn((width,)))

    def forward(self, image, mask_ratio):
        x = self.patch_embed(image)
        x += einops.repeat(
            self.pos_embed.pe_mat.to(x.device), pattern="l d -> b l d", b=x.size(0),
        )[:, : x.size(1), :]
        unmasked_token, mask = self.patch_shuffle(x, mask_ratio=mask_ratio)
        x = self.tf_block(unmasked_token)
        x = self.norm(x)

        new_token = einops.repeat(
            self.mask_token, pattern="w -> b c w", b=x.size(0), c=self.cell_size ** 2,
        ).clone()
        new_token[:, ~mask, :] = x
        return new_token, mask


class Decoder(nn.Module):
    """
    "Each mask token is a shared, learned vector that indicates the presence of a missing
    patch to be predicted. We add positional embeddings to all tokens in this full set;
    without this, mask tokens would have no information about their location in the image.
    The decoder has another series of Transformer blocks."
    "Our MAE reconstructs the input by predicting the pixel values for each masked patch.
    Each element in the decoder’s output is a vector of pixel values representing a patch.
    The last layer of the decoder is a linear projection whose number of output channels
    equals the number of pixel values in a patch. The decoder’s output is reshaped to form
    a reconstructed image."
    "The decoder is applied to this full list (with positional embeddings added)."
    """
    def __init__(
        self,
        img_size,
        patch_size,
        depth,
        width,
        n_heads,
        n_pixel_values=256,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.img_size = img_size
        self.n_pixel_values = n_pixel_values
        self.cell_size = img_size // patch_size

        self.pos_embed = PositionalEmbedding(embed_dim=width)
        self.tf_block = TransformerBlocks(
            depth=depth,
            width=width,
            n_heads=n_heads,
        )
        self.proj = nn.Linear(width, (patch_size ** 2) * 3)

    def forward(self, x):
        x += einops.repeat(
            self.pos_embed.pe_mat.to(x.device), pattern="l d -> b l d", b=x.size(0),
        )[:, : x.size(1), :]
        x = self.tf_block(x)
        x = self.proj(x)
        return einops.rearrange(
            x,
            pattern="b (i1 i2) (p1 p2 c) -> b c (i1 p1) (i2 p2)",
            p1=self.patch_size,
            p2=self.patch_size,
            i1=self.cell_size,
            # c=3,
        )


class MAE(nn.Module):
    """
    "As the MAE encoder and decoder have different width, we adopt a linear projection
    layer after the encoder to match it."
    "Our loss function computes the mean squared error (MSE) between the reconstructed and
    original images in the pixel space. We compute the loss only on masked patches,
    similar to BERT."
    """
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        enc_depth=12,
        enc_width=768,
        enc_n_heads=12,
        dec_depth=2,
        dec_width=128,
        dec_n_heads=2,
    ):
        super().__init__()

        self.patch_size = patch_size

        assert img_size % patch_size == 0, (
            "`img_size` must be divisible by `patch_size`!",
        )
        self.n_cells = img_size // patch_size

        self.encoder = Encoder(
            img_size=img_size,
            patch_size=patch_size,
            depth=enc_depth,
            width=enc_width,
            n_heads=enc_n_heads,
        )
        self.proj = nn.Linear(enc_width, dec_width)
        self.decoder = Decoder(
            img_size=img_size,
            patch_size=patch_size,
            depth=dec_depth,
            width=dec_width,
            n_heads=dec_n_heads,
        )

    def forward(self, image, mask_ratio=0.75):
        x, mask = self.encoder(image, mask_ratio=mask_ratio)
        x = self.proj(x)
        x = self.decoder(x)
        return x, mask

    def upsample_mask(self, mask, batch_size, device):
        up_mask = torch.repeat_interleave(
            torch.repeat_interleave(
                mask.view(self.n_cells, self.n_cells),
                repeats=self.patch_size,
                dim=0,
            ),
            repeats=self.patch_size,
            dim=1,
        ).to(device)
        return einops.repeat(
            up_mask, "h w -> b c h w", b=batch_size, c=3,
        )

    def get_loss(self, image, mask_ratio=0.75):
        out, mask = self(image, mask_ratio=mask_ratio)
        up_mask = self.upsample_mask(
            mask, batch_size=image.size(0), device=image.device,
        )
        return torch.mean(
            up_mask * F.mse_loss(out, image, reduction="none"),
        )

    def reconstruct(self, image, mask_ratio=0.75):
        out, mask = self(image, mask_ratio=mask_ratio)
        up_mask = self.upsample_mask(
            mask, batch_size=image.size(0), device=image.device,
        )
        masked_image = torch.where(up_mask, torch.full_like(image, fill_value=0), image)
        recon_image = torch.where(up_mask, out, image)
        return masked_image, recon_image


class MAEForCls(MAE):
    """
    "We extract features from the encoder output for finetuning and linear probing.
    As ViT has a class token, to adapt to this design, in our MAE pre-training
    we append an auxiliary dummy token to the encoder input. This token will be treated
    as the class token for training the classifier in linear probing and fine-tuning.
    Our MAE works similarly well without this token (with average pooling)."
    """
    def __init__(
        self,
        n_classes,
        img_size=256,
        patch_size=16,
        enc_depth=12,
        enc_width=768,
        enc_n_heads=12,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            enc_depth=enc_depth,
            enc_width=enc_width,
            enc_n_heads=enc_n_heads,
        )
        self.proj = nn.Linear(enc_width, n_classes)

    def forward(self, image):
        x, _ = self.encoder(image, mask_ratio=0)
        x = torch.mean(x, dim=1)
        return self.proj(x)


# "We do not use color jittering, drop path, or gradient clip."
# "We use the linear lr scaling rule [20]: lr = base lr batchsize / 256. End-to-end fine-tuning."
if __name__ == "__main__":
    patch_size = 16
    img_size = 256
    enc_depth = 12
    enc_width = 768
    enc_n_heads = 12
    dec_depth = 2
    dec_width = 128
    dec_n_heads = 2

    image = torch.randn((4, 3, img_size, img_size))
    # model = MAE(
    #     img_size=img_size,
    #     patch_size=patch_size,
    #     enc_depth=enc_depth,
    #     enc_width=enc_width,
    #     enc_n_heads=enc_n_heads,
    #     dec_depth=dec_depth,
    #     dec_width=dec_width,
    #     dec_n_heads=dec_n_heads,
    # )
    # loss = model.get_loss(image)
    # print(loss)

    model = MAEForCls(
        n_classes=1000,
        img_size=img_size,
        patch_size=patch_size,
        enc_depth=enc_depth,
        enc_width=enc_width,
        enc_n_heads=enc_n_heads,
    )
    out = model(image)
    print(out.shape)
