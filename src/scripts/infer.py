from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_config import ModelConfig
from configs.inference_config import InferenceConfig

from models.encoder_decoder import EncoderDecoder
from custom_tokenizers.image_tokenizer import ImageTokenizer
from custom_tokenizers.string_tokenizer import StringTokenizer

from rendering.geometry import Geometry
from rendering.rasterizer import WuRasterizer

from inference.inference import Inference


logger = logging.getLogger("string_art.inference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a string-art configuration from an input image "
            "using a trained Transformer checkpoint."
        )
    )

    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Path to the model configuration JSON file.",
    )

    parser.add_argument(
        "--inference-config",
        type=Path,
        required=True,
        help="Path to the inference configuration JSON file.",
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the input grayscale image.",
    )

    return parser.parse_args()


def resolve_device(device_setting: str) -> torch.device:
    """Resolve the configured device, including automatic selection."""

    setting = device_setting.strip().lower()

    if setting == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(setting)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but CUDA is not available."
        )

    if (
        device.type == "mps"
        and (
            not hasattr(torch.backends, "mps")
            or not torch.backends.mps.is_available()
        )
    ):
        raise RuntimeError(
            "MPS was requested, but MPS is not available."
        )

    return device


def validate_paths(args: argparse.Namespace) -> None:
    """Validate configuration and input paths before initialization."""

    required_files = {
        "Model configuration": args.model_config,
        "Inference configuration": args.inference_config,
        "Input image": args.image,
    }

    for description, path in required_files.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"{description} does not exist: {path}"
            )


def build_inference(
    model_config: ModelConfig,
    inference_config: InferenceConfig,
    device: torch.device,
) -> Inference:
    """Construct the model and its inference dependencies."""

    # The model owns its image tokenizer and image encoder.
    model = EncoderDecoder(model_config).to(device)

    string_tokenizer = StringTokenizer(model_config)

    geometry = Geometry(model_config)
    rasterizer = WuRasterizer(model_config, geometry)

    return Inference(
        model=model,
        string_tokenizer=string_tokenizer,
        rasterizer=rasterizer,
        config=inference_config,
        device=device,
    )


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    try:
        validate_paths(args)

        # Load configurations.
        model_config = ModelConfig.from_json(str(args.model_config))
        inference_config = InferenceConfig.from_json(
            str(args.inference_config)
        )

        checkpoint_path = Path(inference_config.checkpoint_path)

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint does not exist: {checkpoint_path}"
            )

        # Resolve and report the execution device.
        device = resolve_device(inference_config.device)
        logger.info("Using device: %s", device)
        logger.info("Input image: %s", args.image)
        logger.info("Checkpoint: %s", checkpoint_path)

        # Construct inference pipeline and load trained weights.
        inference = build_inference(
            model_config=model_config,
            inference_config=inference_config,
            device=device,
        )

        inference.load_checkpoint(checkpoint_path)

        # Run prediction.
        result = inference.predict_from_file(args.image)

        strings = result["strings"]
        tokens = result["tokens"]

        logger.info("Inference completed successfully.")
        logger.info("Generated strings: %d", len(strings))
        logger.info("Generated token count: %d", len(tokens))

        print(f"\nGenerated strings: {len(strings)}")
        print("\nEndpoint-token pairs:")

        for index, (endpoint_0, endpoint_1) in enumerate(strings):
            print(
                f"  String {index:03d}: "
                f"({endpoint_0}, {endpoint_1})"
            )

        if result.get("rendered_image_path") is not None:
            print(
                "\nReconstructed image saved to: "
                f"{result['rendered_image_path']}"
            )

        return 0

    except (FileNotFoundError, ValueError, RuntimeError, TypeError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
