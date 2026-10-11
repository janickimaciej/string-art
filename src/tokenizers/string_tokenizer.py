from abc import ABC, abstractmethod
from typing import Sequence

from configs.model_config import ModelConfig


class IStringTokenizer(ABC):
    """Converts string configurations to and from endpoint-token sequences."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    def encode(
        self,
        strings: Sequence[tuple[int, int]],
        add_special_tokens: bool = True,
    ) -> list[int]:
        """
        Encode strings as a flat endpoint-token sequence.

        Each tuple contains two endpoint tokens, not pin indices.

        Canonicalize endpoints and sort strings deterministically.
        Optionally prepend BOS and append EOS.
        """

    @abstractmethod
    def decode(
        self,
        tokens: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> list[tuple[int, int]]:
        """
        Decode endpoint tokens into string pairs.

        Validate token IDs and complete endpoint pairs.
        Canonicalize endpoints and sort strings deterministically.
        """
