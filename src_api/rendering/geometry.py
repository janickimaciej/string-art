from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class Point:
    """2D point in image coordinates."""

    x: float
    y: float


@dataclass(frozen=True)
class Endpoint:
    """A string endpoint identified by pin and connection side."""

    pin: int
    side: int


@dataclass(frozen=True)
class String:
    """A string connecting two endpoints."""

    start: Endpoint
    end: Endpoint


class PinGeometry:
    """Converts discrete pin indices into physical/image coordinates."""

    def __init__(
        self,
        num_pins: int,
        center: tuple[float, float],
        radius: float,
    ) -> None:
        """
        Initialize the circular pin geometry.

        Args:
            num_pins: Number of pins around the circular frame.
            center: (x, y) coordinates of the circle center.
            radius: Radius of the circle in image coordinates.
        """
        ...

    def pin_position(
        self,
        pin: int,
    ) -> Point:
        """
        Return the image-space position of a pin.

        Args:
            pin: Clockwise pin index.

        Returns:
            Position of the requested pin.
        """
        ...

    def endpoint_position(
        self,
        endpoint: Endpoint,
    ) -> Point:
        """
        Return the image-space position of an endpoint.

        Args:
            endpoint: Pin and connection-side information.

        Returns:
            Position of the requested string endpoint.
        """
        ...

    def string_endpoints(
        self,
        string: String,
    ) -> tuple[Point, Point]:
        """
        Convert a string's discrete endpoints into image-space points.

        Args:
            string: String connecting two endpoints.

        Returns:
            Start and end points of the string.
        """
        ...
