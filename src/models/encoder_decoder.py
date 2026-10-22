from abc import ABC, abstractmethod

import torch
from torch import nn

from configs.model_config import ModelConfig
from custom_tokenizers.image_tokenizer import ImageTokenizer
from models.image_encoder import ImageEncoder
from models.decoder import GPT2Decoder


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

class EncoderDecoder(IEncoderDecoder):
    """
    Complete image-to-endpoint-token sequence model.

    Pipeline:
        Grayscale image
            -> ImageTokenizer
            -> ImageEncoder
            -> GPT2Decoder with cross-attention
            -> Endpoint-token logits or generated sequence

    The decoder predicts endpoint tokens only. Geometry decoding,
    string canonicalization, and rasterization belong to other modules.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        feature_channels: tuple[int, ...] = (32, 64, 128, 256),
        token_grid_size: tuple[int, int] = (16, 16),
    ) -> None:
        super().__init__(config)

        self.image_tokenizer = ImageTokenizer(
            config,
            feature_channels=feature_channels,
            token_grid_size=token_grid_size,
        )

        self.image_encoder = ImageEncoder(config)

        self.decoder = GPT2Decoder(
            config,
        )

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Convert grayscale images into contextualized image features.

        Args:
            images: [B, 1, H, W] or [1, H, W].

        Returns:
            encoder_hidden_states: [B, N, D].
        """
        image_tokens = self.image_tokenizer.encode(images)

        encoder_hidden_states = self.image_encoder(image_tokens)

        return encoder_hidden_states

    def forward(
        self,
        images: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Teacher-forced forward pass.

        Args:
            images: [B, 1, H, W] grayscale images.
            decoder_input_ids: [B, T] shifted decoder input sequences.
            attention_mask: Optional [B, T], 1 for valid tokens.

        Returns:
            logits: [B, T, vocab_size].
        """
        if decoder_input_ids.ndim != 2:
            raise ValueError(
                "decoder_input_ids must have shape [B, T], "
                f"got {tuple(decoder_input_ids.shape)}."
            )

        encoder_hidden_states = self.encode_image(images)

        if encoder_hidden_states.shape[0] != decoder_input_ids.shape[0]:
            raise ValueError(
                "Image batch size and decoder input batch size must match."
            )

        logits = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )

        return logits

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        max_strings: int | None = None,
        temperature: float = 1.0,
        greedy: bool = True,
    ) -> torch.Tensor:
        """
        Generate endpoint-token sequences conditioned on images.

        Args:
            images: [B, 1, H, W] grayscale images.
            max_strings: Maximum number of generated strings.
            temperature: Sampling temperature for non-greedy decoding.
            greedy: Use argmax decoding if True.

        Returns:
            sequences: [B, T], including BOS and generated EOS.
        """
        was_training = self.training
        self.eval()

        try:
            encoder_hidden_states = self.encode_image(images)

            sequences = self.decoder.generate(
                encoder_hidden_states=encoder_hidden_states,
                max_strings=max_strings,
                temperature=temperature,
                greedy=greedy,
            )

        finally:
            self.train(was_training)

        return sequences
