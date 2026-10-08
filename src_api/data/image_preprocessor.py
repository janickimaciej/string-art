from torch import Tensor


class ImagePreprocessor:
    """Preprocesses input images into model-ready tensors."""

    def __init__(
        self,
        image_size: int,
        grayscale: bool = True,
        normalize: bool = True,
    ) -> None:
        """
        Initialize the image preprocessor.

        Args:
            image_size: Target height and width of the processed image.
            grayscale: Whether to convert input images to grayscale.
            normalize: Whether to normalize pixel values for the model.
        """
        ...

    def __call__(
        self,
        image: object,
    ) -> Tensor:
        """
        Preprocess a loaded image.

        Args:
            image: Input image, typically a PIL Image.

        Returns:
            Image tensor with shape [channels, height, width].
        """
        ...
