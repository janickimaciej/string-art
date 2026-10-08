from torch import Tensor

from rendering.differentiable_rasterizer import DifferentiableRasterizer
from rendering.geometry import String

from .image_loss import ImageLoss


class StringArtReward:
    """Computes reinforcement-learning rewards for generated string configurations."""

    def __init__(
        self,
        rasterizer: DifferentiableRasterizer,
        image_loss: ImageLoss,
        reward_scale: float = 1.0,
    ) -> None:
        """
        Initialize the string-art reward.

        Args:
            rasterizer: Differentiable renderer used to produce images.
            image_loss: Image reconstruction loss used to evaluate renderings.
            reward_scale: Scalar multiplier applied to the resulting reward.
        """
        ...

    def compute(
        self,
        strings: list[String],
        target: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Render a string configuration and calculate its reward.

        Args:
            strings: Generated strings for one target image.
            target: Target image to reproduce.

        Returns:
            Tuple containing:
                - reward: Scalar reward, where higher is better.
                - loss: Image reconstruction loss.
        """
        ...

    def compute_batch(
        self,
        strings: list[list[String]],
        targets: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Calculate rewards for a batch of generated string configurations.

        Args:
            strings: Generated string configurations, one per target image.
            targets: Target images with shape [batch, 1, height, width].

        Returns:
            Tuple containing:
                - rewards: Per-example rewards.
                - losses: Per-example image reconstruction losses.
        """
        ...
