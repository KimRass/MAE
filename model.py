# References:
    # https://github.com/KimRass/ViT/blob/main/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from utils import denorm


DROP_PROB = 0.1
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, hidden_size, drop_prob=DROP_PROB):
        super().__init__()

        self.patch_size = patch_size

        in_feats = (patch_size ** 2) * 3
        self.norm1 = nn.LayerNorm(in_feats)
        self.proj = nn.Linear(in_feats, hidden_size)
        self.drop = nn.Dropout(drop_prob)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = rearrange(
            x,
            pattern="b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        x = self.norm1(x)
        x = self.proj(x)
        x = self.drop(x)
        x = self.norm2(x)
        return x


class MSA(nn.Module):
    def __init__(self, hidden_size, n_heads, drop_prob=DROP_PROB):
        super().__init__()

        assert (
            hidden_size % n_heads == 0,
            "`hidden_size` must be divisible by `n_heads`!",
        )

        self.head_size = hidden_size // n_heads
        self.n_heads = n_heads

        self.qkv_proj = nn.Linear(hidden_size, 3 * n_heads * self.head_size, bias=False)
        self.drop = nn.Dropout(drop_prob)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _get_attention_score(self, q, k):
        attn_score = torch.einsum("bhnd,bhmd->bhnm", q, k)
        return attn_score

    def forward(self, x):
        q, k, v = torch.split(
            self.qkv_proj(x), split_size_or_sections=self.n_heads * self.head_size, dim=2,
        )
        q = rearrange(q, pattern="b n (h d) -> b h n d", h=self.n_heads, d=self.head_size)
        k = rearrange(k, pattern="b n (h d) -> b h n d", h=self.n_heads, d=self.head_size)
        v = rearrange(v, pattern="b n (h d) -> b h n d", h=self.n_heads, d=self.head_size)
        attn_score = self._get_attention_score(q=q, k=k)
        attn_weight = F.softmax(attn_score / (self.head_size ** 0.5), dim=3)
        x = torch.einsum("bhnm,bhmd->bhnd", attn_weight, v)
        x = rearrange(x, pattern="b h n d -> b n (h d)")
        x = self.out_proj(x)
        x = self.drop(x)
        return x


class SkipConnection(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, sublayer):
        skip = x.clone()
        x = self.norm(x)
        x = sublayer(x)
        x += skip
        return x


class MLP(nn.Module):
    def __init__(self, hidden_size, mlp_size):
        super().__init__()

        self.proj1 = nn.Linear(hidden_size, mlp_size)
        self.drop1 = nn.Dropout(0.1)
        self.proj2 = nn.Linear(mlp_size, hidden_size)
        self.drop2 = nn.Dropout(0.1)

    def forward(self, x):
        x = self.proj1(x)
        x = F.gelu(x)
        x = self.drop1(x)
        x = self.proj2(x)
        x = F.gelu(x)
        x = self.drop2(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, mlp_size, n_heads):
        super().__init__()

        self.self_attn = MSA(hidden_size=hidden_size, n_heads=n_heads)
        self.self_attn_resid = SkipConnection(hidden_size=hidden_size)
        self.mlp = MLP(hidden_size=hidden_size, mlp_size=mlp_size)
        self.mlp_resid = SkipConnection(hidden_size=hidden_size)

    def forward(self, x):
        x = self.self_attn_resid(x=x, sublayer=self.self_attn)
        x = self.mlp_resid(x=x, sublayer=self.mlp)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, n_layers, hidden_size, mlp_size, n_heads):
        super().__init__()

        self.tf_layers = nn.ModuleList(
            [TransformerLayer(hidden_size=hidden_size, mlp_size=mlp_size, n_heads=n_heads)
                for _ in range(n_layers)]
        )

    def forward(self, x):
        for layer in self.tf_layers:
            x = layer(x)
        return x


class Tokenizer(nn.Module):
    """
    "First we generate a token for every input patch (by linear projection with an added positional embedding)."
    "our encoder embeds patches by a linear projection with added positional embeddings."
    """
    def __init__(self, img_size, patch_size):
        super().__init__()

        assert (
            img_size % patch_size == 0,
            "`img_size` must be divisible by `patch_size`!",
        )

        self.cell_size = img_size // patch_size

        self.patch_embed = PatchEmbedding(patch_size=patch_size, hidden_size=hidden_size)
        self.pos_embed = nn.Parameter(torch.randn((1, self.cell_size ** 2, hidden_size)))

    def forward(self, x):
        x = self.patch_embed(image)
        x += self.pos_embed
        return x


class Shuffler(nn.Module):
    """
    "Next we randomly shuffle the list of tokens and remove the last portion of the list, based on the masking ratio. This process produces a small subset of tokens for the encoder and is equivalent to sampling patches without replacement."
    """
    def __init__(self, mask_ratio=0.75):
        super().__init__()

        self.mask_ratio = mask_ratio

    def forward(self, x):
        n_tokens = x.size(1)
        shuffled_indices = torch.randperm(n_tokens)
        mask_indices = shuffled_indices[-int(n_tokens * self.mask_ratio):]
        mask = torch.isin(torch.arange(n_tokens), mask_indices)
        return x[:, ~mask, :], mask


class Encoder(nn.Module):
    """
    "Our encoder is a ViT but applied only on visible, unmasked patches. Just as in a standard ViT, then processes the resulting set via a series of Transformer blocks. However, our encoder only operates on a small subset (e.g., 25%) of the full set. Masked patches are removed; no mask tokens are used."
    "After encoding, we append a list of mask tokens to the list of encoded patches, and
    unshuffle this full list (inverting the random shuffle operation) to align all tokens
    with their targets."
    """
    def __init__(
        self,
        img_size,
        patch_size,
        mask_ratio,
        n_layers,
        hidden_size,
        mlp_size,
        n_heads,
    ):
        super().__init__()

        self.tokenizer = Tokenizer(img_size=img_size, patch_size=patch_size)
        self.shuffler = Shuffler(mask_ratio=mask_ratio)
        self.tf_block = TransformerBlock(
            n_layers=n_layers,
            hidden_size=hidden_size,
            mlp_size=mlp_size,
            n_heads=n_heads,
        )
        self.mask_token = nn.Parameter(torch.randn((hidden_size,)))

    def forward(self, image):
        token = self.tokenizer(image)
        unmasked_token, mask = self.shuffler(token)
        x = self.tf_block(unmasked_token)
        
        y = self.mask_token[None, None, :].repeat(
            x.size(0), self.tokenizer.cell_size ** 2, 1,
        )
        y[:, ~mask, :] = x
        return y, mask


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
    "Our loss function computes the mean squared error (MSE) between the reconstructed and
    original images in the pixel space. We compute the loss only on masked patches,
    similar to BERT."
    "The decoder is applied to this full list (with positional embeddings added)."
    """
    def __init__(
        self,
        img_size,
        patch_size,
        n_layers,
        hidden_size,
        mlp_size,
        n_heads,
        n_pixel_values=256,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.img_size = img_size
        self.n_pixel_values = n_pixel_values

        self.cell_size = img_size // patch_size

        self.pos_embed = nn.Parameter(torch.randn((1, self.cell_size ** 2, hidden_size)))
        self.tf_block = TransformerBlock(
            n_layers=n_layers,
            hidden_size=hidden_size,
            mlp_size=mlp_size,
            n_heads=n_heads,
        )
        # self.proj = nn.Linear(hidden_size, (patch_size ** 2) * 3)
        self.proj = nn.Linear(hidden_size, (patch_size ** 2) * 3 * n_pixel_values)

    def forward(self, x):
        x += self.pos_embed
        x = self.tf_block(x)
        x = self.proj(x)
        x = rearrange(
            x,
            pattern="b (i1 i2) (p1 p2 c n) -> b c (i1 p1) (i2 p2) n",
            p1=self.patch_size,
            p2=self.patch_size,
            i1=self.img_size // self.patch_size,
            c=3,
        )
        return x


class MAE(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        n_layers,
        hidden_size,
        mlp_size,
        n_heads,
        mask_ratio=0.75,
    ):
        super().__init__()

        self.patch_size = patch_size

        self.n_cells = img_size // patch_size

        self.encoder = Encoder(
            img_size=img_size,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            n_layers=n_layers,
            hidden_size=hidden_size,
            mlp_size=mlp_size,
            n_heads=n_heads,
        )
        self.decoder = Decoder(
            img_size=img_size,
            patch_size=patch_size,
            n_layers=n_layers,
            hidden_size=hidden_size,
            mlp_size=mlp_size,
            n_heads=n_heads,
        )

    def forward(self, image):
        x, mask = self.encoder(image)
        x = self.decoder(x)
        return x, mask

    @staticmethod
    def image_to_gt(image):
        gt = denorm(image)
        gt.clip_(min=0, max=1)
        gt *= 255
        return gt.long()

    def get_loss(self, image):
        out, mask = self(image)

        mask2 = torch.repeat_interleave(
            torch.repeat_interleave(
                mask.view(self.n_cells, self.n_cells),
                repeats=self.patch_size,
                dim=0,
            ),
            repeats=self.patch_size,
            dim=1,
        )
        gt = self.image_to_gt(image)
        masked_out = out[:, :, mask2, :]
        masked_gt = gt[:, :, mask2]    
        return F.cross_entropy(
            masked_out.view(-1, self.decoder.n_pixel_values),
            masked_gt.view(-1),
            reduction="mean",
        )


if __name__ == "__main__":
    patch_size = 16
    hidden_size = 1024
    mlp_size = hidden_size * 4
    img_size = 256
    n_layers = 24
    n_heads = 8
    image = torch.randn((4, 3, img_size, img_size))
    model = MAE(
        img_size,
        patch_size,
        n_layers,
        hidden_size,
        mlp_size,
        n_heads,
    )
    mask, out = model(image)
    loss = model.get_loss(image)
    print(loss)
