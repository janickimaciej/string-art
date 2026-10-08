import torch
from torch import Tensor, nn


class ImageTokenizer(nn.Module):
    """Converts input images into spatial feature tokens for the encoder."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        patch_size: int,
    ) -> None:
        """
        Initialize the convolutional image tokenizer.

        Args:
            in_channels: Number of channels in the input image.
            hidden_dim: Dimension of each output image token.
            patch_size: Spatial size and stride of each convolutional patch.
        """
        ...

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        """
        Convert images into a sequence of image tokens.

        Args:
            images: Input images with shape [batch, channels, height, width].

        Returns:
            Image tokens with shape [batch, num_tokens, hidden_dim].
        """
        ...
