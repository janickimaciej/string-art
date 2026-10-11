from abc import ABC, abstractmethod

import torch
from torch import nn

from configs.model_config import ModelConfig


class IImageEncoder(nn.Module, ABC):
    """Encodes spatial image tokens for decoder cross-attention."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, image_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_tokens: [B, N, D] spatial feature tokens.

        Returns:
            encoder_hidden_states: [B, N, D], contextualized
            image tokens used as decoder cross-attention memory.
        """
        raise NotImplementedError


class ImageEncoder(IImageEncoder):
    """
    Transformer encoder over spatial image tokens.

    Input:
        image_tokens: [B, N, D]

    Output:
        encoder_hidden_states: [B, N, D]

    The encoder contextualizes the spatial features produced by
    ImageTokenizer. Its output can be passed directly to a decoder
    configured with encoder-decoder cross-attention.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

        if config.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if config.num_heads <= 0:
            raise ValueError("num_heads must be positive.")

        if config.hidden_dim % config.num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads; "
                f"received hidden_dim={config.hidden_dim}, "
                f"num_heads={config.num_heads}."
            )

        if config.encoder_layers < 1:
            raise ValueError("encoder_layers must be at least 1.")

        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.encoder_layers,
            norm=nn.LayerNorm(config.hidden_dim),
        )

    def forward(self, image_tokens: torch.Tensor) -> torch.Tensor:
        """
        Contextualize spatial image tokens.

        Args:
            image_tokens:
                Floating-point tensor of shape [B, N, D].

        Returns:
            Tensor of shape [B, N, D], suitable as decoder
            cross-attention memory.
        """

        if not isinstance(image_tokens, torch.Tensor):
            raise TypeError(
                "image_tokens must be a torch.Tensor, "
                f"got {type(image_tokens).__name__}."
            )

        if image_tokens.ndim != 3:
            raise ValueError(
                "Expected image_tokens with shape [B, N, D], "
                f"received {tuple(image_tokens.shape)}."
            )

        batch_size, num_tokens, hidden_dim = image_tokens.shape

        if batch_size < 1:
            raise ValueError("The image-token batch must not be empty.")

        if num_tokens < 1:
            raise ValueError("The image-token sequence must not be empty.")

        if hidden_dim != self.config.hidden_dim:
            raise ValueError(
                f"Expected token dimension D={self.config.hidden_dim}, "
                f"received D={hidden_dim}."
            )

        if not image_tokens.is_floating_point():
            raise TypeError(
                "image_tokens must have a floating-point dtype; "
                f"received {image_tokens.dtype}."
            )

        if not torch.isfinite(image_tokens).all():
            raise ValueError("image_tokens contains NaN or infinite values.")

        # [B, N, D] -> [B, N, D]
        encoder_hidden_states = self.transformer_encoder(image_tokens)

        return encoder_hidden_states
