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
        Return the 256 pin centers in image coordinates.

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
    def decode_endpoint(self, token: int) -> tuple[float, float]:
        """
        Convert an endpoint token into its attachment coordinates.

        Token encoding:
            pin_index = token // 2
            side = token % 2

        Returns:
            (x, y) in target-image pixel coordinates.
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
