from torch import Tensor, nn

from .geometry import PinGeometry, String


class DifferentiableRasterizer(nn.Module):
    """Differentiably renders strings into a grayscale image."""

    def __init__(
        self,
        geometry: PinGeometry,
        image_size: int,
        line_width: float,
        background_value: float = 0.0,
    ) -> None:
        """
        Initialize the differentiable rasterizer.

        Args:
            geometry: Pin geometry used to convert endpoints into coordinates.
            image_size: Height and width of the output image.
            line_width: Width of rendered strings in image-space units.
            background_value: Pixel value used for the empty background.
        """
        ...

    def forward(
        self,
        strings: list[String],
    ) -> Tensor:
        """
        Render strings into a differentiable grayscale image.

        Args:
            strings: Strings to render.

        Returns:
            Rendered image with shape [1, image_size, image_size].
        """
        ...

    def render_batch(
        self,
        strings: list[list[String]],
    ) -> Tensor:
        """
        Render multiple string configurations.

        Args:
            strings: List of string configurations, one per batch element.

        Returns:
            Rendered images with shape
            [batch, 1, image_size, image_size].
        """
        ...
