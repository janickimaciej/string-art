from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch

from configs.model_config import ModelConfig
from rendering.IGeometry import IGeometry


class IRasterizer(ABC):
    """Renders string configurations into grayscale images."""

    def __init__(
        self,
        config: ModelConfig,
        geometry: IGeometry,
    ) -> None:
        self.config = config
        self.geometry = geometry

    @abstractmethod
    def render(
        self,
        strings: Sequence[tuple[int, int]],
    ) -> torch.Tensor:
        """
        Render a string configuration.

        Args:
            strings: Sequence of (endpoint_token_0, endpoint_token_1).

        Returns:
            Grayscale image Tensor [H, W], with floating-point
            intensities in [0, 1]:
                0 = white background
                1 = maximum darkness

        Requirements:
            - Use Xiaolin Wu's anti-aliased line algorithm.
            - Render at config.supersampling_factor resolution.
            - Use a string width of one high-resolution pixel.
            - Accumulate overlapping string darkness and clamp to [0, 1].
            - Downsample to config.image_height x config.image_width.
            - Use geometry to decode side-specific attachment points.
        """
        raise NotImplementedError

    @abstractmethod
    def save_image(
        self,
        image: torch.Tensor,
        path: str,
    ) -> None:
        """Save a rendered grayscale image to disk."""
        raise NotImplementedError
