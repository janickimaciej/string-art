from abc import ABC, abstractmethod

import torch

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
