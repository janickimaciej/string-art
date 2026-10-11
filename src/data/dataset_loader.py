from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from configs.model_config import ModelConfig


class IDatasetLoader(Dataset, ABC):
    """Reads previously generated image-token pairs from disk."""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        model_config: ModelConfig,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.model_config = model_config

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of saved examples in this split."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Load one saved example.

        Returns:
            {
                "image": Tensor[C, H, W],
                "tokens": Tensor[T],
            }

        Requirements:
            - Read only persisted dataset files.
            - Validate dataset metadata and configuration compatibility.
            - Preserve BOS and EOS.
            - Do not pad sequences here; padding belongs in collate.py.
        """
        raise NotImplementedError
