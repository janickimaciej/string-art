from abc import ABC, abstractmethod

from configs.model_config import ModelConfig
from configs.data_generation_config import DataGenerationConfig


class IStringGenerator(ABC):
    """Generates random, canonical string configurations."""

    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataGenerationConfig,
    ) -> None:
        self.model_config = model_config
        self.data_config = data_config

    @abstractmethod
    def generate(self) -> list[tuple[int, int]]:
        """
        Generate a random string configuration.

        Returns:
            A list of (endpoint_token_0, endpoint_token_1) pairs.

        Requirements:
            - Each endpoint token is in [0, 511].
            - Canonicalize endpoints and sort strings deterministically.
            - Respect configured string-count limits.
            - Respect duplicate and degenerate-string constraints.
        """
        raise NotImplementedError
