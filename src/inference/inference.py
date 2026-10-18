from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch

from configs.inference_config import InferenceConfig
from models.IEncoderDecoder import IEncoderDecoder
from custom_tokenizers.IImageTokenizer import IImageTokenizer
from custom_tokenizers.IStringTokenizer import IStringTokenizer
from rendering.IRasterizer import IRasterizer


class IInference(ABC):
    """Generates string configurations from input images."""

    def __init__(
        self,
        model: IEncoderDecoder,
        image_tokenizer: IImageTokenizer,
        string_tokenizer: IStringTokenizer,
        rasterizer: IRasterizer,
        config: InferenceConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.image_tokenizer = image_tokenizer
        self.string_tokenizer = string_tokenizer
        self.rasterizer = rasterizer
        self.config = config
        self.device = device

    @abstractmethod
    def load_checkpoint(self, path: str | Path) -> None:
        """Load trained model parameters from a checkpoint."""
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        image: torch.Tensor,
    ) -> dict[str, Any]:
        """
        Generate a string configuration for one image.

        Args:
            image: Grayscale image [H, W] or [1, H, W].

        Returns:
            A dictionary containing:
                - tokens: generated sequence including BOS/EOS when present
                - strings: decoded endpoint-token pairs
                - rendered_image: reconstructed image, if requested

        Requirements:
            - Use autoregressive generation.
            - Respect max_strings, temperature, and greedy settings.
            - Run without gradient computation.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_from_file(
        self,
        image_path: str | Path,
    ) -> dict[str, Any]:
        """Load an image from disk and run prediction."""
        raise NotImplementedError
