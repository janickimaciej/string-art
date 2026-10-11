from abc import ABC, abstractmethod

import torch

from configs.model_config import ModelConfig


class IGeometry(ABC):
    """Defines pin geometry and string attachment points."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    def get_pin_centers(self) -> torch.Tensor:
        """
        Return pin centers in image coordinates.

        Returns:
            Tensor [num_pins, 2], with (x, y) coordinates
            in target-image pixel space.
        """
        raise NotImplementedError

    @abstractmethod
    def get_attachment_points(self) -> torch.Tensor:
        """
        Return the two attachment points for every pin.

        Returns:
            Tensor [num_pins, 2, 2].

            Indexing:
                points[pin_index, connection_side] -> (x, y)

        The two points for each pin must be separated by
        exactly config.pin_diameter pixels.
        """
        raise NotImplementedError

    @abstractmethod
    def decode_string(
        self,
        endpoint_0: int,
        endpoint_1: int,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Convert two endpoint tokens into string endpoint coordinates.
        """
        raise NotImplementedError


class Geometry(IGeometry):
    """
    Circular frame geometry with two attachment points per pin.

    Pin indices increase clockwise, starting at angle zero on the
    positive x-axis. Image coordinates use x increasing rightward and
    y increasing downward.

    Endpoint tokens:
        token = 2 * pin_index + side
        side  = 0 or 1

    The attachment points are offset along the pin's tangent by
    +/- pin_diameter / 2, so their separation equals pin_diameter.
    """

    def __init__(
        self,
        config: ModelConfig,
        frame_center: tuple[float, float] | None = None,
    ) -> None:
        super().__init__(config)

        if config.num_pins <= 0:
            raise ValueError("num_pins must be positive.")

        if config.image_height <= 0 or config.image_width <= 0:
            raise ValueError("Image dimensions must be positive.")

        if config.frame_radius < 0:
            raise ValueError("frame_radius must be non-negative.")

        if config.pin_diameter < 0:
            raise ValueError("pin_diameter must be non-negative.")

        if frame_center is None:
            frame_center = (
                (config.image_width - 1) / 2.0,
                (config.image_height - 1) / 2.0,
            )

        self.frame_center = (
            float(frame_center[0]),
            float(frame_center[1]),
        )

        # Precompute geometry once. Both public geometry methods use
        # the same tensors to ensure consistent coordinates.
        self._pin_centers, self._attachment_points = (
            self._compute_geometry()
        )

    def _compute_geometry(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute pin centers and attachment points in float64."""
        num_pins = self.config.num_pins
        radius = float(self.config.frame_radius)
        diameter = float(self.config.pin_diameter)
        cx, cy = self.frame_center

        theta = (
            2.0
            * torch.pi
            * torch.arange(num_pins, dtype=torch.float64)
            / num_pins
        )

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # Pin centers. Positive image y points downward, so increasing
        # theta naturally produces clockwise indexing in image space.
        centers = torch.stack(
            (
                cx + radius * cos_theta,
                cy + radius * sin_theta,
            ),
            dim=-1,
        )

        # Unit tangent at each pin.
        tangents = torch.stack(
            (
                -sin_theta,
                cos_theta,
            ),
            dim=-1,
        )

        # Side 0 lies in the negative tangent direction; side 1 lies
        # in the positive tangent direction.
        offsets = (diameter / 2.0) * tangents

        attachment_points = torch.stack(
            (
                centers - offsets,
                centers + offsets,
            ),
            dim=1,
        )

        return centers, attachment_points

    def get_pin_centers(self) -> torch.Tensor:
        """
        Return pin centers in target-image pixel coordinates.

        Returns:
            Tensor of shape [num_pins, 2], ordered by pin index.
        """
        return self._pin_centers.clone()

    def get_attachment_points(self) -> torch.Tensor:
        """
        Return side-specific attachment coordinates.

        Returns:
            Tensor of shape [num_pins, 2, 2].
        """
        return self._attachment_points.clone()

    def decode_string(
        self,
        endpoint_0: int,
        endpoint_1: int,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Decode two endpoint tokens into attachment coordinates.

        This method resolves both tokens for the complete string.
        Future geometry subclasses may override it if attachment
        positions depend on the other endpoint.

        Args:
            endpoint_0: First endpoint token.
            endpoint_1: Second endpoint token.

        Returns:
            ((x0, y0), (x1, y1)) in target-image pixel coordinates.
        """
        token_count = 2 * self.config.num_pins

        for token in (endpoint_0, endpoint_1):
            if not isinstance(token, int):
                raise TypeError(
                    f"Endpoint tokens must be integers; got {type(token).__name__}."
                )

            if not 0 <= token < token_count:
                raise ValueError(
                    f"Endpoint token {token} is out of range. "
                    f"Expected a value in [0, {token_count - 1}]."
                )

        pin_0, side_0 = divmod(endpoint_0, 2)
        pin_1, side_1 = divmod(endpoint_1, 2)

        point_0 = self._attachment_points[pin_0, side_0]
        point_1 = self._attachment_points[pin_1, side_1]

        return (
            (point_0[0].item(), point_0[1].item()),
            (point_1[0].item(), point_1[1].item()),
        )
