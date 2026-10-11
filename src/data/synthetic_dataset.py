from abc import ABC, abstractmethod

from configs.model_config import ModelConfig
from configs.data_generation_config import DataGenerationConfig
from tokenizers.string_tokenizer import IStringTokenizer
from data.string_generator import IStringGenerator


class ISyntheticDataset(ABC):
    """Generates and persists synthetic image-token datasets."""

    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataGenerationConfig,
        string_generator: IStringGenerator,
        string_tokenizer: IStringTokenizer,
    ) -> None:
        self.model_config = model_config
        self.data_config = data_config
        self.string_generator = string_generator
        self.string_tokenizer = string_tokenizer

    @abstractmethod
    def generate_and_save(self) -> None:
        """
        Generate and save train, validation, and test splits.

        Each example contains:
            image: grayscale image at configured target resolution.
            tokens: integer sequence [BOS, ..., EOS].

        Requirements:
            - Render using the configured Wu rasterizer.
            - Save image-token pairs in indexed or sharded files.
            - Save metadata describing the generation configuration.
            - Use independent, reproducible random seeds per split.
            - Do not overwrite existing datasets without explicit permission.
        """
        raise NotImplementedError
