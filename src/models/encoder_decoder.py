from abc import ABC, abstractmethod

import torch
from torch import nn

from configs.model_config import ModelConfig


class IEncoderDecoder(nn.Module, ABC):
    """Complete image-to-string-sequence model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Convert images into contextualized image features.

        Args:
            images: [B, 1, H, W].

        Returns:
            encoder_hidden_states: [B, N, D].
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        images: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Teacher-forced forward pass for supervised training.

        Args:
            images: [B, 1, H, W].
            decoder_input_ids: [B, T] shifted token sequences.
            attention_mask: [B, T] decoder padding mask.

        Returns:
            logits: [B, T, vocab_size].
        """
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        images: torch.Tensor,
        max_strings: int | None = None,
        temperature: float = 1.0,
        greedy: bool = True,
    ) -> torch.Tensor:
        """
        Generate endpoint-token sequences conditioned on images.

        Returns:
            sequences: [B, T], including BOS and EOS when generated.
        """
        raise NotImplementedError
