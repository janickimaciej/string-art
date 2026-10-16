from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from PIL import Image

from configs.model_config import ModelConfig
from rendering.geometry import IGeometry, Geometry


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


@triton.jit
def _rasterize_lines_kernel(
    lines,
    canvas,
    width: tl.constexpr,
    height: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Rasterize one antialiased line per Triton program."""
    line_id = tl.program_id(0)

    x0 = tl.load(lines + line_id * 4 + 0)
    y0 = tl.load(lines + line_id * 4 + 1)
    x1 = tl.load(lines + line_id * 4 + 2)
    y1 = tl.load(lines + line_id * 4 + 3)

    # Liang-Barsky clipping against pixel-center bounds.
    dx = x1 - x0
    dy = y1 - y0
    t_enter = 0.0
    t_exit = 1.0

    for i in tl.static_range(4):
        if i == 0:
            p = -dx
            q = x0
        elif i == 1:
            p = dx
            q = (width - 1.0) - x0
        elif i == 2:
            p = -dy
            q = y0
        else:
            p = dy
            q = (height - 1.0) - y0

        if p == 0.0:
            if q < 0.0:
                return
        else:
            t = q / p
            if p < 0.0:
                t_enter = tl.maximum(t_enter, t)
            else:
                t_exit = tl.minimum(t_exit, t)

            if t_enter > t_exit:
                return

    # Preserve the original start point before updating coordinates.
    orig_x0 = x0
    orig_y0 = y0

    x0 = orig_x0 + t_enter * dx
    y0 = orig_y0 + t_enter * dy
    x1 = orig_x0 + t_exit * dx
    y1 = orig_y0 + t_exit * dy

    # Wu's algorithm operates on shallow lines.
    steep = tl.abs(y1 - y0) > tl.abs(x1 - x0)

    if steep:
        tmp = x0
        x0 = y0
        y0 = tmp
        tmp = x1
        x1 = y1
        y1 = tmp

    if x0 > x1:
        tmp = x0
        x0 = x1
        x1 = tmp
        tmp = y0
        y0 = y1
        y1 = tmp

    dx = x1 - x0
    dy = y1 - y0

    # Degenerate segment: render one pixel.
    if tl.abs(dx) < 1.0e-12:
        px = tl.floor(x0 + 0.5).to(tl.int32)
        py = tl.floor(y0 + 0.5).to(tl.int32)

        if steep:
            tmp = px
            px = py
            py = tmp

        if (
            (px >= 0)
            & (px < width)
            & (py >= 0)
            & (py < height)
        ):
            tl.atomic_add(
                canvas + py * width + px,
                1.0,
                sem="relaxed",
            )
        return

    gradient = dy / dx

    # First endpoint, including its fractional coverage.
    x_first = tl.floor(x0 + 0.5).to(tl.int32)
    y_first = y0 + gradient * (x_first.to(tl.float32) - x0)
    gap_first = 1.0 - (
        (x0 + 0.5) - tl.floor(x0 + 0.5)
    )

    iy_first = tl.floor(y_first).to(tl.int32)
    f_first = y_first - tl.floor(y_first)

    c0 = (1.0 - f_first) * gap_first
    c1 = f_first * gap_first

    if steep:
        px = iy_first
        py = x_first
    else:
        px = x_first
        py = iy_first

    if (
        (px >= 0) & (px < width)
        & (py >= 0) & (py < height)
        & (c0 > 0.0)
    ):
        tl.atomic_add(
            canvas + py * width + px, c0, sem="relaxed"
        )

    if steep:
        px = iy_first + 1
        py = x_first
    else:
        px = x_first
        py = iy_first + 1

    if (
        (px >= 0) & (px < width)
        & (py >= 0) & (py < height)
        & (c1 > 0.0)
    ):
        tl.atomic_add(
            canvas + py * width + px, c1, sem="relaxed"
        )

    # Second endpoint.
    x_last = tl.floor(x1 + 0.5).to(tl.int32)
    y_last = y1 + gradient * (x_last.to(tl.float32) - x1)
    gap_last = (x1 + 0.5) - tl.floor(x1 + 0.5)

    iy_last = tl.floor(y_last).to(tl.int32)
    f_last = y_last - tl.floor(y_last)

    c0 = (1.0 - f_last) * gap_last
    c1 = f_last * gap_last

    if steep:
        px = iy_last
        py = x_last
    else:
        px = x_last
        py = iy_last

    if (
        (px >= 0) & (px < width)
        & (py >= 0) & (py < height)
        & (c0 > 0.0)
    ):
        tl.atomic_add(
            canvas + py * width + px, c0, sem="relaxed"
        )

    if steep:
        px = iy_last + 1
        py = x_last
    else:
        px = x_last
        py = iy_last + 1

    if (
        (px >= 0) & (px < width)
        & (py >= 0) & (py < height)
        & (c1 > 0.0)
    ):
        tl.atomic_add(
            canvas + py * width + px, c1, sem="relaxed"
        )

    # Interior pixels, processed in parallel blocks.
    start = x_first + 1
    stop = x_last
    offsets = tl.arange(0, BLOCK)
    offset = 0

    while offset < stop - start:
        xs = start + offset + offsets
        mask = xs < stop

        ys = y_first + gradient * (xs.to(tl.float32) - x_first)
        iy = tl.floor(ys).to(tl.int32)
        frac = ys - tl.floor(ys)

        cov0 = 1.0 - frac
        cov1 = frac

        if steep:
            px0 = iy
            py0 = xs
            px1 = iy + 1
            py1 = xs
        else:
            px0 = xs
            py0 = iy
            px1 = xs
            py1 = iy + 1

        valid0 = (
            mask
            & (px0 >= 0) & (px0 < width)
            & (py0 >= 0) & (py0 < height)
            & (cov0 > 0.0)
        )
        valid1 = (
            mask
            & (px1 >= 0) & (px1 < width)
            & (py1 >= 0) & (py1 < height)
            & (cov1 > 0.0)
        )

        tl.atomic_add(
            canvas + py0 * width + px0,
            cov0,
            sem="relaxed",
            mask=valid0,
        )
        tl.atomic_add(
            canvas + py1 * width + px1,
            cov1,
            sem="relaxed",
            mask=valid1,
        )

        offset += BLOCK


class WuRasterizer(IRasterizer):
    """
    GPU-accelerated antialiased line rasterizer.

    Uses a Wu-style coverage kernel, additive atomic accumulation,
    saturation, and GPU-side area downsampling.

    Returns a CPU float32 tensor to preserve the original API's
    output-device behavior.
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

        if not torch.cuda.is_available():
            raise RuntimeError("WuRasterizer requires a CUDA GPU.")

        self.device = torch.device("cuda")
        self.scale = int(config.supersampling_factor)
        self.high_height = config.image_height * self.scale
        self.high_width = config.image_width * self.scale

        # Reuse the canvas across sequential images.
        self._canvas = torch.zeros(
            (self.high_height, self.high_width),
            dtype=torch.float32,
            device=self.device,
        )

        self._attachment_points_gpu = None

        # Vectorized decoding is valid when Geometry's standard
        # implementation of decode_string is in use.
        self._vectorized_geometry = (
            isinstance(geometry, Geometry)
            and type(geometry).decode_string is Geometry.decode_string
        )

        if self._vectorized_geometry:
            self._attachment_points_gpu = (
                geometry.get_attachment_points()
                .to(device=self.device, dtype=torch.float32)
                .contiguous()
            )

    @torch.no_grad()
    def render(
        self,
        strings: Sequence[tuple[int, int]],
    ) -> torch.Tensor:
        # Validate pairs and retain the original int conversion behavior.
        endpoints = []
        for pair in strings:
            if len(pair) != 2:
                raise ValueError(
                    "Each string must contain exactly two endpoint tokens."
                )
            endpoints.append((int(pair[0]), int(pair[1])))

        self._canvas.zero_()

        if endpoints:
            tokens = torch.tensor(
                endpoints,
                dtype=torch.int64,
                device=self.device,
            )

            if self._vectorized_geometry:
                token_count = 2 * self.config.num_pins
                if (
                    (tokens < 0).any().item()
                    or (tokens >= token_count).any().item()
                ):
                    raise ValueError(
                        f"Endpoint tokens must be in [0, {token_count - 1}]."
                    )

                # [N, 2, 2]: coordinates for both endpoints of each line.
                points = self._attachment_points_gpu[
                    tokens // 2, tokens % 2
                ]
                lines = points.reshape(-1, 4).contiguous()
            else:
                # Preserve custom geometry semantics: decode each
                # complete pair on the CPU before transferring coordinates.
                coordinates = []
                for a, b in endpoints:
                    (x0, y0), (x1, y1) = self.geometry.decode_string(a, b)
                    coordinates.append((x0, y0, x1, y1))

                lines = torch.tensor(
                    coordinates,
                    dtype=torch.float32,
                    device=self.device,
                )

            # Validate coordinates before launching the GPU kernel.
            if not torch.isfinite(lines).all().item():
                raise ValueError("Line coordinates must be finite.")

            # Convert target-image coordinates to high-resolution pixels.
            lines = lines * self.scale
            lines = lines.contiguous()

            _rasterize_lines_kernel[(len(endpoints),)](
                lines,
                self._canvas,
                self.high_width,
                self.high_height,
                BLOCK=256,
                num_warps=4,
            )

        # Accumulation and downsampling stay on the GPU.
        self._canvas.clamp_(0.0, 1.0)

        if self.scale > 1:
            image = F.avg_pool2d(
                self._canvas[None, None],
                kernel_size=self.scale,
                stride=self.scale,
            )[0, 0]
        else:
            image = self._canvas

        return image.clamp_(0.0, 1.0).cpu().contiguous()

    def save_image(
        self,
        image: torch.Tensor,
        path: str,
    ) -> None:
        if image.ndim != 2:
            raise ValueError(
                f"Expected a 2D grayscale image, got shape {tuple(image.shape)}."
            )

        expected = (
            self.config.image_height,
            self.config.image_width,
        )
        if tuple(image.shape) != expected:
            raise ValueError(
                f"Expected image shape {expected}, got {tuple(image.shape)}."
            )

        image = image.detach().to(
            device="cpu", dtype=torch.float32
        ).clamp(0.0, 1.0)

        pixels = (
            (1.0 - image) * 255.0
        ).round().to(torch.uint8).numpy()

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(output_path)
