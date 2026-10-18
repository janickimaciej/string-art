import argparse
from pathlib import Path
import torch

from configs.model_config import ModelConfig
from rendering.rasterizer import WuRasterizer
from rendering.geometry import Geometry

assert torch.cuda.is_available()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and persist synthetic string-art image-token "
            "datasets using the configured geometry and rasterizer."
        )
    )

    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Path to the model configuration JSON file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_config = ModelConfig.from_json(
        str(args.model_config)
    )

    geometry = Geometry(model_config)

    rasterizer = WuRasterizer(model_config, geometry)

    strings = [
        (0, 100),
        (10, 200),
        (20, 300),
        (0, 100),  # Check repeated-line accumulation.
    ]

    image = rasterizer.render(strings)

    assert image.shape == (
        model_config.image_height,
        model_config.image_width,
    )
    assert image.dtype == torch.float32
    assert image.device.type == "cpu"
    assert torch.isfinite(image).all()
    assert image.min() >= 0.0
    assert image.max() <= 1.0

    rasterizer.save_image(image, "artifacts/test.png")


if __name__ == "__main__":
    main()
