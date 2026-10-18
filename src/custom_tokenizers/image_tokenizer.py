from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from configs.model_config import ModelConfig


class IImageTokenizer(ABC):
    """Converts grayscale images into spatial feature tokens."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """
        Convert grayscale images into spatial feature tokens.

        Input:
            image: [B, 1, H, W] or [1, H, W]

        Output:
            tokens: [B, N, D]

        Where:
            B = batch size
            N = number of spatial feature tokens
            D = config.hidden_dim

        Responsibilities:
            - Validate image dimensions and grayscale channels.
            - Normalize image intensities consistently.
            - Extract convolutional features.
            - Flatten spatial dimensions into a token sequence.
            - Project features to config.hidden_dim.
            - Add spatial positional embeddings.
        """


class ConvBlock(nn.Module):
    """A convolutional feature extraction block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 2,
    ) -> None:
        super().__init__()

        # Select a valid number of groups for GroupNorm.
        num_groups = min(8, out_channels)
        while out_channels % num_groups != 0:
            num_groups -= 1

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ImageTokenizer(nn.Module, IImageTokenizer):
    """
    CNN-based grayscale image tokenizer.

    Architecture:
        Grayscale image
        -> CNN feature extractor
        -> Adaptive spatial pooling
        -> Flatten spatial dimensions
        -> Linear feature projection
        -> Learned spatial positional embeddings

    For the default configuration, the output contains 256 spatial
    tokens, each with config.hidden_dim features.

    Intensity convention:
        0.0 = white
        1.0 = maximum darkness

    Supported input representations:
        - uint8 images in [0, 255]
        - floating-point images in [0, 1]
        - floating-point images in [0, 255]
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        feature_channels: tuple[int, ...] = (32, 64, 128, 256),
        token_grid_size: tuple[int, int] = (16, 16),
    ) -> None:
        nn.Module.__init__(self)
        IImageTokenizer.__init__(self, config)

        if len(feature_channels) != 4:
            raise ValueError(
                "feature_channels must contain exactly four channel sizes."
            )

        if any(c <= 0 for c in feature_channels):
            raise ValueError("All feature channel sizes must be positive.")

        grid_height, grid_width = token_grid_size
        if grid_height <= 0 or grid_width <= 0:
            raise ValueError("Token grid dimensions must be positive.")

        if config.image_height <= 0 or config.image_width <= 0:
            raise ValueError("Configured image dimensions must be positive.")

        self.token_grid_size = token_grid_size
        self.num_spatial_tokens = grid_height * grid_width

        # Four convolutional stages progressively extract visual features.
        channels = (1, *feature_channels)
        self.feature_extractor = nn.Sequential(
            *[
                ConvBlock(
                    channels[i],
                    channels[i + 1],
                    stride=2,
                )
                for i in range(len(feature_channels))
            ]
        )

        # Fix the spatial token count independently of the intermediate
        # feature map dimensions.
        self.spatial_pool = nn.AdaptiveAvgPool2d(token_grid_size)

        # Map CNN features to the shared Transformer hidden dimension.
        self.feature_projection = nn.Linear(
            feature_channels[-1],
            config.hidden_dim,
        )

        # Each spatial location has its own learned positional embedding.
        self.spatial_position_embeddings = nn.Parameter(
            torch.empty(
                1,
                self.num_spatial_tokens,
                config.hidden_dim,
            )
        )

        self.dropout = nn.Dropout(config.dropout)

        nn.init.trunc_normal_(
            self.spatial_position_embeddings,
            mean=0.0,
            std=0.02,
        )

    def _validate_and_normalize(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Validate image shape and convert intensities to float [0, 1]."""

        if not isinstance(image, torch.Tensor):
            raise TypeError(
                f"Expected a torch.Tensor, got {type(image).__name__}."
            )

        if image.ndim == 3:
            # [1, H, W] -> [1, 1, H, W]
            if image.shape[0] != 1:
                raise ValueError(
                    "A 3D image must have shape [1, H, W]; "
                    f"received {tuple(image.shape)}."
                )
            image = image.unsqueeze(0)

        elif image.ndim != 4:
            raise ValueError(
                "Expected image shape [B, 1, H, W] or [1, H, W]; "
                f"received {tuple(image.shape)}."
            )

        if image.shape[0] < 1:
            raise ValueError("The image batch must not be empty.")

        if image.shape[1] != 1:
            raise ValueError(
                "Only single-channel grayscale images are supported; "
                f"received {image.shape[1]} channels."
            )

        expected_size = (
            self.config.image_height,
            self.config.image_width,
        )
        actual_size = tuple(image.shape[-2:])

        if actual_size != expected_size:
            raise ValueError(
                f"Expected image dimensions HxW={expected_size}, "
                f"received HxW={actual_size}."
            )

        if image.dtype == torch.uint8:
            # Preserve the original grayscale intensity convention.
            image = image.to(dtype=torch.float32).div(255.0)

        elif image.is_floating_point():
            if not torch.isfinite(image).all():
                raise ValueError("Image contains NaN or infinite values.")

            image = image.to(dtype=torch.float32)

            # Accept both normalized floating-point images and images
            # represented using the original 8-bit intensity range.
            minimum = image.amin().item()
            maximum = image.amax().item()

            if minimum < 0.0 or maximum > 255.0:
                raise ValueError(
                    "Floating-point image intensities must lie in "
                    "[0, 1] or [0, 255]. "
                    f"Observed range [{minimum}, {maximum}]."
                )

            if maximum > 1.0:
                image = image.div(255.0)

        else:
            raise TypeError(
                "Image dtype must be torch.uint8 or a floating-point "
                f"dtype; received {image.dtype}."
            )

        return image

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """
        Convert a grayscale image batch into spatial feature tokens.

        Args:
            image:
                [B, 1, H, W] or [1, H, W].

        Returns:
            tokens:
                [B, N, D], where:
                    B = batch size
                    N = token_grid_size[0] * token_grid_size[1]
                    D = config.hidden_dim
        """

        image = self._validate_and_normalize(image)

        # [B, 1, H, W] -> [B, C, H', W']
        features = self.feature_extractor(image)

        # [B, C, H', W'] -> [B, C, Gh, Gw]
        features = self.spatial_pool(features)

        # [B, C, Gh, Gw] -> [B, Gh*Gw, C]
        features = features.flatten(start_dim=2).transpose(1, 2)

        # [B, N, C] -> [B, N, D]
        tokens = self.feature_projection(features)

        # Add learned spatial position information.
        tokens = tokens + self.spatial_position_embeddings

        return self.dropout(tokens)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Allow the tokenizer to be called as a standard PyTorch module."""
        return self.encode(image)
