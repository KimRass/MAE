import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from vit import ViT, PatchEmbedding, TransformerBlock

# "After encoding, we append a list of mask tokens to the list of encoded patches, and unshuffle this full list (inverting the random shuffle operation) to align all tokens with their targets. The decoder is applied to this full list (with positional embeddings added). As noted, no sparse operations are needed. This simple implementation"


class Tokenizer(nn.Module):
    """
    "First we generate a token for every input patch (by linear projection with an added positional embedding)."
    "our encoder embeds patches by a linear projection with added positional embeddings."
    """
    def __init__(self, img_size, patch_size):
        super().__init__()

        assert img_size % patch_size == 0, "`img_size` must be divisible by `patch_size`!"

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
    def __init__(self, n_tokens, mask_ratio=0.75):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.n_tokens = n_tokens

    def forward(self, x):
        shuffled_indices = torch.randperm(self.n_tokens)
        mask_indices = shuffled_indices[: int(self.n_tokens * (1 - self.mask_ratio))]
        return mask_indices, x[:, ~mask_indices, :]


class Encoder(nn.Module):
    """
    "Our encoder is a ViT but applied only on visible, unmasked patches. Just as in a standard ViT, then processes the resulting set via a series of Transformer blocks. However, our encoder only operates on a small subset (e.g., 25%) of the full set. Masked patches are removed; no mask tokens are used."
    """
    def __init__(
        self,
        img_size,
        patch_size,
        n_tokens,
        mask_ratio,
        n_layers,
        hidden_size,
        mlp_size,
        n_heads,
    ):
        super().__init__()

        self.tokenizer = Tokenizer(img_size=img_size, patch_size=patch_size)
        self.shuffler = Shuffler(n_tokens=n_tokens, mask_ratio=mask_ratio)
        self.tf_block = TransformerBlock(
            n_layers=n_layers,
            hidden_size=hidden_size,
            mlp_size=mlp_size,
            n_heads=n_heads,
        )
        self.mask_token = nn.Parameter(torch.randn((hidden_size,)))

    def forward(self, image):
        token = self.tokenizer(image)
        mask_indices, masked_token = self.shuffler(token)
        x = self.tf_block(masked_token)
        
        y = self.mask_token[None, None, :].repeat(
            x.size(0), self.tokenizer.cell_size ** 2, 1,
        )
        y[:, mask_indices, :] = x
        return y


class Decoder(nn.Module):
    """
    "Each mask token is a shared, learned vector that indicates the presence of a missing
    patch to be predicted. We add positional embeddings to all tokens in this full set;
    without this, mask tokens would have no information about their location in the image.
    The decoder has another series of Transformer blocks."
    "Our MAE reconstructs the input by predicting the pixel values for each masked patch.
    Each element in the decoder’s output is a vector of pixel values representing a patch.
    The last layer of the decoder is a linear projection whose number of output channels
    equals the number of pixel values in a patch. The decoder’s output is reshaped to form a reconstructed image. Our loss function computes the mean squared error (MSE) between the reconstructed and original images in the pixel space. We compute the loss only on masked patches, similar to BERT."
    """
    def __init__(
        self,
        img_size,
        patch_size,
        n_layers,
        hidden_size,
        mlp_size,
        n_heads,
    ):
        super().__init__()

        self.cell_size = img_size // patch_size

        self.pos_embed = nn.Parameter(torch.randn((1, self.cell_size ** 2, hidden_size)))
        self.tf_block = TransformerBlock(
            n_layers=n_layers,
            hidden_size=hidden_size,
            mlp_size=mlp_size,
            n_heads=n_heads,
        )
        dim = (patch_size ** 2) * 3
        self.proj = nn.Linear(hidden_size, dim)

    def forward(self, x):
        x += self.pos_embed
        x = self.tf_block(x)
        x = self.proj(x)
        x = rearrange(
            x,
            pattern="b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            p1=patch_size,
            p2=patch_size,
            h=patch_size,
            w=patch_size,
        )
        return x


if __name__ == "__main__":
    patch_size = 16
    hidden_size = 192
    img_size = 256
    n_layers = 12
    mlp_size = 3072
    n_heads = 12
    image = torch.randn((4, 3, img_size, img_size))

    encoder = Encoder(
        img_size=img_size,
        patch_size=patch_size,
        n_tokens=256,
        mask_ratio=0.75,
        n_layers=n_layers,
        hidden_size=hidden_size,
        mlp_size=mlp_size,
        n_heads=n_heads,
    )
    decoder = Decoder(
        img_size=img_size,
        patch_size=patch_size,
        n_layers=n_layers,
        hidden_size=hidden_size,
        mlp_size=mlp_size,
        n_heads=n_heads,
    )
    x = encoder(image)
    x = decoder(x)
    x.shape
