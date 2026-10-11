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
