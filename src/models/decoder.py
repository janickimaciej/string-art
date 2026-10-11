from abc import ABC, abstractmethod

import torch
from torch import nn

from configs.model_config import ModelConfig


class IDecoder(nn.Module, ABC):
    """Autoregressive endpoint-token decoder with image cross-attention."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Teacher-forced forward pass.

        Args:
            input_ids: [B, T] decoder input token IDs.
            encoder_hidden_states: [B, N, D] image encoder output.
            attention_mask: [B, T], 1 for valid decoder tokens.
            encoder_attention_mask: [B, N], 1 for valid image tokens.

        Returns:
            logits: [B, T, vocab_size].
        """
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        encoder_hidden_states: torch.Tensor,
        max_strings: int | None = None,
        temperature: float = 1.0,
        greedy: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressively generate endpoint tokens.

        Start with BOS and stop at EOS or the configured length limit.

        Returns:
            sequences: [B, T], including BOS and EOS when generated.
        """
        raise NotImplementedError
