from pathlib import Path

from torch import Tensor

from models.encoder_decoder import EncoderDecoder
from rendering.geometry import String
from tokenizers.image_tokenizer import ImageTokenizer
from tokenizers.string_tokenizer import StringTokenizer


class Inference:
    """Runs inference to generate strings for target images."""

    def __init__(
        self,
        model: EncoderDecoder,
        image_tokenizer: ImageTokenizer,
        string_tokenizer: StringTokenizer,
        device: str,
        max_sequence_length: int,
        temperature: float = 1.0,
    ) -> None:
        """
        Initialize the inference pipeline.

        Args:
            model: Trained image-to-string encoder-decoder model.
            image_tokenizer: Converts input images into encoder tokens.
            string_tokenizer: Converts generated token IDs into strings.
            device: Device on which inference is performed.
            max_sequence_length: Maximum number of tokens generated.
            temperature: Sampling temperature used during generation.
        """
        ...

    def generate_tokens(
        self,
        image: Tensor,
        do_sample: bool = True,
    ) -> list[int]:
        """
        Generate a token sequence for one target image.

        Args:
            image: Preprocessed image with shape
                [channels, height, width].
            do_sample: Whether to sample from the token distribution.
                If False, use greedy decoding.

        Returns:
            Generated decoder token IDs.
        """
        ...

    def generate_strings(
        self,
        image: Tensor,
        do_sample: bool = True,
    ) -> list[String]:
        """
        Generate a string configuration for one target image.

        Args:
            image: Preprocessed target image.
            do_sample: Whether to sample during autoregressive generation.

        Returns:
            Generated strings.
        """
        ...

    def save_strings(
        self,
        strings: list[String],
        path: str | Path,
    ) -> None:
        """
        Save generated strings to disk.

        Args:
            strings: Generated string configuration.
            path: Output path for the serialized string configuration.
        """
        ...
