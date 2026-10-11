from abc import ABC, abstractmethod
from typing import Any

import torch

from configs.model_config import ModelConfig


class ICollate(ABC):
    """Combines dataset examples into a training batch."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    def __call__(
        self,
        examples: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        """
        Collate examples into a padded batch.

        Input example:
            {
                "image": Tensor[C, H, W],
                "tokens": Tensor[T],
            }

        Output:
            {
                "images": Tensor[B, C, H, W],
                "decoder_input_ids": Tensor[B, T_max - 1],
                "labels": Tensor[B, T_max - 1],
                "attention_mask": Tensor[B, T_max - 1],
            }

        Decoder inputs are tokens[:-1].
        Labels are tokens[1:].

        Padding positions in labels must use -100 so that
        PyTorch cross-entropy ignores them.
        """
        raise NotImplementedError
