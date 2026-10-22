from __future__ import annotations

import argparse
import sys
from pathlib import Path

from configs.model_config import ModelConfig
from configs.data_generation_config import DataGenerationConfig

from data.string_generator import StringGenerator
from data.synthetic_dataset import SyntheticDataset

from rendering.geometry import Geometry
from rendering.rasterizer import WuRasterizer
from custom_tokenizers.string_tokenizer import StringTokenizer


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

    parser.add_argument(
        "--data-generation-config",
        type=Path,
        required=True,
        help="Path to the data-generation configuration JSON file.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacing an existing dataset directory.",
    )

    return parser.parse_args()


def validate_config_paths(
    model_config_path: Path,
    data_generation_config_path: Path,
) -> None:
    """Validate configuration files before loading them."""
    for path in (model_config_path, data_generation_config_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Configuration file does not exist: {path}"
            )


def main() -> int:
    args = parse_args()

    try:
        validate_config_paths(
            args.model_config,
            args.data_generation_config,
        )

        # Load configuration exclusively from the supplied JSON files.
        model_config = ModelConfig.from_json(
            str(args.model_config)
        )

        data_generation_config = DataGenerationConfig.from_json(
            str(args.data_generation_config)
        )

        output_dir = Path(data_generation_config.output_dir)

        print("String Art synthetic dataset generation")
        print("---------------------------------------")
        print(f"Model config:  {args.model_config}")
        print(f"Data config:   {args.data_generation_config}")
        print(f"Output dir:    {output_dir}")
        print(
            "Split sizes:   "
            f"train={data_generation_config.train_samples}, "
            f"validation={data_generation_config.validation_samples}, "
            f"test={data_generation_config.test_samples}"
        )
        print(
            "String limits: "
            f"{data_generation_config.min_strings}.."
            f"{min(data_generation_config.max_strings, model_config.max_strings)}"
        )
        print(f"Base seed:     {data_generation_config.seed}")
        print()

        # Initialize the shared geometry and rendering dependencies.
        geometry = Geometry(model_config)
        rasterizer = WuRasterizer(model_config, geometry)
        tokenizer = StringTokenizer(model_config)

        generator = StringGenerator(
            model_config=model_config,
            data_generation_config=data_generation_config,
        )

        dataset = SyntheticDataset(
            model_config=model_config,
            data_generation_config=data_generation_config,
            string_generator=generator,
            string_tokenizer=tokenizer,
            rasterizer=rasterizer,
            allow_overwrite=args.overwrite,
        )

        dataset.generate_and_save()

        print()
        print("Dataset generation completed successfully.")
        print(f"Dataset location: {output_dir.resolve()}")

        return 0

    except (FileNotFoundError, FileExistsError, ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    except KeyboardInterrupt:
        print(
            "\nDataset generation interrupted by the user.",
            file=sys.stderr,
        )
        return 130

    except Exception as exc:
        print(
            f"Dataset generation failed: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
