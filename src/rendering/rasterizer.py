from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
import math

import torch
import torch.nn.functional as F
from PIL import Image

from configs.model_config import ModelConfig
from rendering.geometry import IGeometry


class IRasterizer(ABC):
    """Renders string configurations into grayscale images."""

    def __init__(
        self,
        config: ModelConfig,
        geometry: IGeometry,
    ) -> None:
        self.config = config
        self.geometry = geometry

    @abstractmethod
    def render(
        self,
        strings: Sequence[tuple[int, int]],
    ) -> torch.Tensor:
        """
        Render a string configuration.

        Args:
            strings: Sequence of (endpoint_token_0, endpoint_token_1).

        Returns:
            Grayscale image Tensor [H, W], with floating-point
            intensities in [0, 1]:
                0 = white background
                1 = maximum darkness

        Requirements:
            - Use Xiaolin Wu's anti-aliased line algorithm.
            - Render at config.supersampling_factor resolution.
            - Use a string width of one high-resolution pixel.
            - Accumulate overlapping string darkness and clamp to [0, 1].
            - Downsample to config.image_height x config.image_width.
            - Use geometry to decode side-specific attachment points.
        """
        raise NotImplementedError

    @abstractmethod
    def save_image(
        self,
        image: torch.Tensor,
        path: str,
    ) -> None:
        """Save a rendered grayscale image to disk."""
        raise NotImplementedError


class WuRasterizer(IRasterizer):
    """
    Deterministic Xiaolin Wu line rasterizer.

    Coordinates returned by IGeometry are in target-image pixel space.
    Lines are rendered at supersampling resolution with a nominal width
    of one high-resolution pixel.

    Intensity convention:
        0.0 = white background
        1.0 = maximum darkness

    Each string contributes anti-aliased coverage additively. The final
    high-resolution image is area-downsampled to the target resolution.
    """

    def __init__(
        self,
        config: ModelConfig,
        geometry: IGeometry,
    ) -> None:
        super().__init__(config, geometry)

        if config.image_height <= 0 or config.image_width <= 0:
            raise ValueError("Image dimensions must be positive.")

        if config.supersampling_factor < 1:
            raise ValueError("supersampling_factor must be >= 1.")

        self.scale = int(config.supersampling_factor)
        self.high_height = config.image_height * self.scale
        self.high_width = config.image_width * self.scale

    @staticmethod
    def _ipart(x: float) -> int:
        return math.floor(x)

    @staticmethod
    def _round(x: float) -> int:
        # Match the conventional Wu algorithm's floor(x + 0.5).
        return math.floor(x + 0.5)

    @staticmethod
    def _fpart(x: float) -> float:
        return x - math.floor(x)

    @staticmethod
    def _rfpart(x: float) -> float:
        return 1.0 - (x - math.floor(x))

    @staticmethod
    def _clip_line(
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        width: int,
        height: int,
    ) -> tuple[float, float, float, float] | None:
        """
        Clip a segment to the raster's pixel-coordinate rectangle using
        Liang-Barsky clipping. This prevents extremely long off-canvas
        segments from requiring unnecessarily long rasterization loops.
        """
        if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
            raise ValueError("Line coordinates must be finite.")

        xmin, ymin = 0.0, 0.0
        xmax, ymax = float(width - 1), float(height - 1)

        dx = x1 - x0
        dy = y1 - y0

        p = (-dx, dx, -dy, dy)
        q = (
            x0 - xmin,
            xmax - x0,
            y0 - ymin,
            ymax - y0,
        )

        t_enter = 0.0
        t_exit = 1.0

        for pi, qi in zip(p, q):
            if pi == 0.0:
                if qi < 0.0:
                    return None
                continue

            t = qi / pi

            if pi < 0.0:
                t_enter = max(t_enter, t)
            else:
                t_exit = min(t_exit, t)

            if t_enter > t_exit:
                return None

        return (
            x0 + t_enter * dx,
            y0 + t_enter * dy,
            x0 + t_exit * dx,
            y0 + t_exit * dy,
        )

    @staticmethod
    def _plot(
        canvas: torch.Tensor,
        x: int,
        y: int,
        coverage: float,
    ) -> None:
        """Accumulate coverage at a valid pixel."""
        if (
            0 <= x < canvas.shape[1]
            and 0 <= y < canvas.shape[0]
            and coverage > 0.0
        ):
            canvas[y, x] += coverage

    def _draw_wu_line(
        self,
        canvas: torch.Tensor,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> None:
        """Draw one anti-aliased segment onto a floating-point canvas."""
        clipped = self._clip_line(
            x0, y0, x1, y1,
            width=self.high_width,
            height=self.high_height,
        )

        if clipped is None:
            return

        x0, y0, x1, y1 = clipped

        # Wu's algorithm is easiest to express for shallow lines.
        # Swap axes for steep lines and swap endpoints to draw left-to-right.
        steep = abs(y1 - y0) > abs(x1 - x0)

        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1

        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        dx = x1 - x0
        dy = y1 - y0

        # A zero-length segment has no well-defined line slope. Render it
        # as a single pixel instead of dividing by zero.
        if abs(dx) < 1e-12:
            x = self._round(x0)
            y = self._round(y0)

            if steep:
                x, y = y, x

            self._plot(canvas, x, y, 1.0)
            return

        gradient = dy / dx

        def plot(x: int, y: int, coverage: float) -> None:
            if steep:
                self._plot(canvas, y, x, coverage)
            else:
                self._plot(canvas, x, y, coverage)

        # First endpoint.
        xend = self._round(x0)
        yend = y0 + gradient * (xend - x0)
        xgap = self._rfpart(x0 + 0.5)
        xpxl1 = xend
        ypxl1 = self._ipart(yend)

        plot(xpxl1, ypxl1, self._rfpart(yend) * xgap)
        plot(xpxl1, ypxl1 + 1, self._fpart(yend) * xgap)

        intery = yend + gradient

        # Second endpoint.
        xend = self._round(x1)
        yend = y1 + gradient * (xend - x1)
        xgap = self._fpart(x1 + 0.5)
        xpxl2 = xend
        ypxl2 = self._ipart(yend)

        plot(xpxl2, ypxl2, self._rfpart(yend) * xgap)
        plot(xpxl2, ypxl2 + 1, self._fpart(yend) * xgap)

        # Interior pixels.
        for x in range(xpxl1 + 1, xpxl2):
            y = self._ipart(intery)

            plot(x, y, self._rfpart(intery))
            plot(x, y + 1, self._fpart(intery))

            intery += gradient

    @torch.no_grad()
    def render(
        self,
        strings: Sequence[tuple[int, int]],
    ) -> torch.Tensor:
        """
        Render endpoint-token pairs into a [H, W] float32 darkness image.

        The geometry implementation is responsible for resolving the
        complete string's attachment coordinates. No assumptions are
        made about circular pin arrangements or endpoint independence.
        """
        # Accumulate at high resolution in floating point.
        high_res = torch.zeros(
            (self.high_height, self.high_width),
            dtype=torch.float32,
            device="cpu",
        )

        for endpoints in strings:
            if len(endpoints) != 2:
                raise ValueError(
                    "Each string must contain exactly two endpoint tokens."
                )

            endpoint_0, endpoint_1 = endpoints

            # Coordinates are in target-image pixel space.
            (x0, y0), (x1, y1) = self.geometry.decode_string(
                int(endpoint_0),
                int(endpoint_1),
            )

            # Transform target-image coordinates into high-resolution space.
            x0 = float(x0) * self.scale
            y0 = float(y0) * self.scale
            x1 = float(x1) * self.scale
            y1 = float(y1) * self.scale

            self._draw_wu_line(high_res, x0, y0, x1, y1)

        # Additive darkness accumulation; intersections saturate at 1.
        high_res.clamp_(0.0, 1.0)

        # Area averaging is a box-filter downsampling operation for
        # integer scale factors. Keep floating-point intensities.
        if self.scale > 1:
            image = F.avg_pool2d(
                high_res.unsqueeze(0).unsqueeze(0),
                kernel_size=self.scale,
                stride=self.scale,
            )[0, 0]
        else:
            image = high_res

        return image.clamp(0.0, 1.0).contiguous()

    def save_image(
        self,
        image: torch.Tensor,
        path: str,
    ) -> None:
        """
        Save a darkness tensor as an 8-bit grayscale image.

        The in-memory convention uses 0 = white and 1 = black, so the
        stored display intensity is inverted to 255 = white and 0 = black.
        """
        if image.ndim != 2:
            raise ValueError(
                f"Expected a 2D grayscale image, got shape {tuple(image.shape)}."
            )

        if tuple(image.shape) != (
            self.config.image_height,
            self.config.image_width,
        ):
            raise ValueError(
                "Image dimensions do not match ModelConfig: "
                f"expected {(self.config.image_height, self.config.image_width)}, "
                f"got {tuple(image.shape)}."
            )

        image = image.detach().to(device="cpu", dtype=torch.float32)
        image = image.clamp(0.0, 1.0)

        # Convert darkness into conventional grayscale brightness.
        pixels = ((1.0 - image) * 255.0).round().to(torch.uint8).numpy()

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        Image.fromarray(pixels, mode="L").save(output_path)
