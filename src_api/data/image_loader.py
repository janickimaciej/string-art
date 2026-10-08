from pathlib import Path

from PIL import Image


class ImageLoader:
    """Loads raw images from disk."""

    def __init__(
        self,
        supported_extensions: tuple[str, ...] = (
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".webp",
        ),
    ) -> None:
        """
        Initialize the image loader.

        Args:
            supported_extensions: File extensions accepted by the loader.
        """
        ...

    def load(
        self,
        path: str | Path,
    ) -> Image.Image:
        """
        Load an image from disk.

        Args:
            path: Path to the image file.

        Returns:
            Loaded PIL image.
        """
        ...
