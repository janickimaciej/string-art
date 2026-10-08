from torch import Tensor


class ImageLoss:
    """Computes image reconstruction error between rendered and target images."""

    def __init__(
        self,
        reduction: str = "mean",
    ) -> None:
        """
        Initialize the image reconstruction loss.

        Args:
            reduction: Reduction applied to per-pixel errors.
                Supported values should include "mean" and "sum".
        """
        ...

    def __call__(
        self,
        rendered: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        Compute image reconstruction loss.

        Args:
            rendered: Rendered images.
            target: Target images with the same shape as rendered.

        Returns:
            Scalar reconstruction loss.
        """
        ...
