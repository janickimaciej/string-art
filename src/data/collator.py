from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from configs.model_config import ModelConfig


class ICollator(ABC):
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


class Collator(ICollator):
    """Collate image-token examples for supervised Transformer training.

    Decoder inputs are tokens[:-1].
    Labels are tokens[1:].
    Decoder inputs are padded with EOS, which is a valid vocabulary ID.
    Labels are padded with -100, the default CrossEntropyLoss ignore index.
    Attention masks use 1 for valid positions and 0 for padding.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

    def _validate_example(
        self,
        example: dict[str, Any],
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "image" not in example or "tokens" not in example:
            raise ValueError(
                f"Example {index} must contain 'image' and 'tokens'."
            )

        image = example["image"]
        tokens = example["tokens"]

        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Example {index}: image must be a Tensor.")

        if not isinstance(tokens, torch.Tensor):
            raise TypeError(f"Example {index}: tokens must be a Tensor.")

        expected_image_shape = (
            1,
            self.config.image_height,
            self.config.image_width,
        )

        if tuple(image.shape) != expected_image_shape:
            raise ValueError(
                f"Example {index}: expected image shape "
                f"{expected_image_shape}, got {tuple(image.shape)}."
            )

        if not image.is_floating_point():
            raise TypeError(
                f"Example {index}: image must have a floating-point dtype."
            )

        if not torch.isfinite(image).all().item():
            raise ValueError(f"Example {index}: image contains NaN or infinity.")

        if image.min().item() < 0.0 or image.max().item() > 1.0:
            raise ValueError(
                f"Example {index}: image intensities must be in [0, 1]."
            )

        if tokens.ndim != 1:
            raise ValueError(
                f"Example {index}: tokens must have shape [T]."
            )

        if tokens.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError(
                f"Example {index}: tokens must have an integer dtype."
            )

        tokens = tokens.to(dtype=torch.long)

        if tokens.numel() < 4:
            raise ValueError(
                f"Example {index}: expected BOS, at least two endpoint "
                "tokens, and EOS."
            )

        if tokens.numel() > self.config.max_sequence_length:
            raise ValueError(
                f"Example {index}: sequence length {tokens.numel()} "
                f"exceeds maximum {self.config.max_sequence_length}."
            )

        if tokens[0].item() != self.config.bos_token_id:
            raise ValueError(f"Example {index}: missing BOS token.")

        if tokens[-1].item() != self.config.eos_token_id:
            raise ValueError(f"Example {index}: missing EOS token.")

        endpoints = tokens[1:-1]

        if endpoints.numel() % 2 != 0:
            raise ValueError(
                f"Example {index}: endpoint tokens must occur in pairs."
            )

        if torch.any(endpoints < 0) or torch.any(
            endpoints >= 2 * self.config.num_pins
        ):
            raise ValueError(
                f"Example {index}: invalid endpoint token ID."
            )

        return image.to(dtype=torch.float32), tokens

    def __call__(
        self,
        examples: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("Cannot collate an empty batch.")

        validated = [
            self._validate_example(example, index)
            for index, example in enumerate(examples)
        ]

        images = torch.stack(
            [image for image, _ in validated],
            dim=0,
        )

        # Shift each complete sequence before padding.
        decoder_sequences = [
            tokens[:-1] for _, tokens in validated
        ]
        label_sequences = [
            tokens[1:] for _, tokens in validated
        ]

        # [B, T_max - 1]. EOS is used only as harmless input padding;
        # attention_mask prevents padded positions from being attended to.
        decoder_input_ids = pad_sequence(
            decoder_sequences,
            batch_first=True,
            padding_value=self.config.eos_token_id,
        )

        # [B, T_max - 1]. -100 is ignored by CrossEntropyLoss.
        labels = pad_sequence(
            label_sequences,
            batch_first=True,
            padding_value=-100,
        )

        # Build the mask from original sequence lengths rather than token
        # values, since EOS is also used for decoder-input padding.
        input_lengths = torch.tensor(
            [sequence.numel() for sequence in decoder_sequences],
            dtype=torch.long,
        )

        positions = torch.arange(
            decoder_input_ids.size(1),
            dtype=torch.long,
        )

        attention_mask = (
            positions.unsqueeze(0) < input_lengths.unsqueeze(1)
        ).to(dtype=torch.long)

        return {
            "images": images,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
