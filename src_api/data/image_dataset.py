from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset

from .image_loader import ImageLoader
from .image_preprocessor import ImagePreprocessor


class ImageDataset(Dataset[dict[str, Tensor | str]]):
    """Dataset providing target images for reinforcement-learning training."""

    def __init__(
        self,
        image_paths: list[str | Path],
        image_loader: ImageLoader,
        image_preprocessor: ImagePreprocessor,
    ) -> None:
        """
        Initialize the image dataset.

        Args:
            image_paths: Paths to images included in the dataset.
            image_loader: Loader used to read images from disk.
            image_preprocessor: Preprocessor used to convert images to tensors.
        """
        ...

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        ...

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Tensor | str]:
        """
        Get one training sample.

        Args:
            index: Dataset index.

        Returns:
            Dictionary containing the preprocessed target image and its identifier.
        """
        ...
