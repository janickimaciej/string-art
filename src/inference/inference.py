from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps

from configs.inference_config import InferenceConfig
from custom_tokenizers.string_tokenizer import IStringTokenizer
from models.encoder_decoder import IEncoderDecoder
from rendering.rasterizer import IRasterizer


class IInference(ABC):
    """Generates string configurations from input images."""

    def __init__(
        self,
        model: IEncoderDecoder,
        string_tokenizer: IStringTokenizer,
        rasterizer: IRasterizer,
        config: InferenceConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.string_tokenizer = string_tokenizer
        self.rasterizer = rasterizer
        self.config = config
        self.device = device

    @abstractmethod
    def load_checkpoint(self, path: str | Path) -> None:
        """Load trained model parameters from a checkpoint."""
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        image: torch.Tensor,
    ) -> dict[str, Any]:
        """
        Generate a string configuration for one image.

        Args:
            image: Grayscale image [H, W] or [1, H, W].

        Returns:
            A dictionary containing:
                - tokens: generated sequence including BOS/EOS when present
                - strings: decoded endpoint-token pairs
                - rendered_image: reconstructed image, if requested

        Requirements:
            - Use autoregressive generation.
            - Respect max_strings, temperature, and greedy settings.
            - Run without gradient computation.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_from_file(
        self,
        image_path: str | Path,
    ) -> dict[str, Any]:
        """Load an image from disk and run prediction."""
        raise NotImplementedError


class Inference(IInference):
    """Generates string configurations from grayscale images.

    The EncoderDecoder owns image tokenization and image encoding.
    This class handles input preparation, checkpoint loading, sequence
    generation, decoding, and optional reconstruction rendering.

    Image intensity convention:
        0.0 = white background
        1.0 = maximum darkness

    Tensor inputs are expected to use this convention. Images loaded from
    ordinary grayscale files are inverted to match it.
    """

    def __init__(
        self,
        model: IEncoderDecoder,
        string_tokenizer: IStringTokenizer,
        rasterizer: IRasterizer,
        config: InferenceConfig,
        device: torch.device,
    ) -> None:
        super().__init__(
            model,
            string_tokenizer,
            rasterizer,
            config,
            device,
        )

        self.model.to(self.device)
        self.model.eval()

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model parameters from a Trainer checkpoint.

        Expected checkpoint format:
            {
                "model_state_dict": ...,
                "optimizer_state_dict": ...,
                ...
            }

        Raw model state dictionaries are also supported.
        """

        checkpoint_path = Path(path)

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint does not exist: {checkpoint_path}"
            )

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            # Compatibility with PyTorch versions without weights_only.
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
            )

        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"Invalid checkpoint format: {checkpoint_path}"
            )

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif checkpoint and all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in checkpoint.items()
        ):
            # Also support torch.save(model.state_dict(), path).
            state_dict = checkpoint
        else:
            raise ValueError(
                "Checkpoint must contain 'model_state_dict' or be a "
                "raw model state dictionary."
            )

        if not isinstance(state_dict, dict):
            raise ValueError(
                "Checkpoint field 'model_state_dict' must be a dictionary."
            )

        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Checkpoint parameters are incompatible with the model: "
                f"{checkpoint_path}. Ensure the model architecture and "
                f"configuration match the training checkpoint."
            ) from exc

        self.model.to(self.device)
        self.model.eval()

    def _image_size(self) -> tuple[int, int]:
        """Return the configured (height, width)."""

        model_config = self.model.config
        height = getattr(model_config, "image_height", None)
        width = getattr(model_config, "image_width", None)

        if height is None or width is None:
            raise ValueError(
                "Model configuration must define image_height and "
                "image_width."
            )

        return int(height), int(width)

    def _prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        """Validate and prepare a tensor for model generation.

        Accepts [H, W] or [1, H, W] and returns [1, 1, H, W].
        """

        if not isinstance(image, torch.Tensor):
            raise TypeError("image must be a torch.Tensor.")

        if image.ndim == 2:
            image = image.unsqueeze(0)
        elif image.ndim != 3 or image.shape[0] != 1:
            raise ValueError(
                "Expected image shape [H, W] or [1, H, W], "
                f"got {tuple(image.shape)}."
            )

        expected_height, expected_width = self._image_size()

        if tuple(image.shape[-2:]) != (
            expected_height,
            expected_width,
        ):
            raise ValueError(
                f"Expected image dimensions "
                f"{expected_height}x{expected_width}, "
                f"got {image.shape[-2]}x{image.shape[-1]}. "
                "Resize the input image to the configured dimensions first."
            )

        if image.is_complex():
            raise ValueError("Complex-valued images are not supported.")

        if image.dtype == torch.bool:
            image = image.float()
        elif not image.is_floating_point():
            # Dataset tensors may be stored as uint8.
            image = image.float() / torch.iinfo(image.dtype).max
        else:
            image = image.float()

        if not torch.isfinite(image).all().item():
            raise ValueError("Input image contains NaN or infinite values.")

        if image.numel() and (
            image.min().item() < 0.0 or image.max().item() > 1.0
        ):
            raise ValueError(
                "Image intensities must be in [0, 1], with 0 representing "
                "white and 1 representing maximum darkness."
            )

        return image.unsqueeze(0).to(self.device)

    def _predict(
        self,
        image: torch.Tensor,
        output_path: Path | None = None,
    ) -> dict[str, Any]:
        """Shared implementation for tensor and file-based prediction."""

        model_image = self._prepare_image(image)

        with torch.inference_mode():
            generated = self.model.generate(
                images=model_image,
                max_strings=self.config.max_strings,
                temperature=self.config.temperature,
                greedy=self.config.greedy,
            )

        if not isinstance(generated, torch.Tensor):
            raise TypeError(
                "model.generate() must return a torch.Tensor."
            )

        if generated.ndim != 2 or generated.shape[0] != 1:
            raise ValueError(
                "model.generate() must return sequences with shape [1, T]; "
                f"got {tuple(generated.shape)}."
            )

        tokens = generated[0].detach().cpu().tolist()

        if any(
            not isinstance(token, int) or token < 0
            for token in tokens
        ):
            raise ValueError(
                "Generated sequence contains invalid token IDs."
            )

        # The tokenizer owns token validation, endpoint-pair validation,
        # canonicalization, sorting, and special-token handling.
        strings = self.string_tokenizer.decode(
            tokens,
            skip_special_tokens=True,
        )

        if len(strings) > self.config.max_strings:
            raise ValueError(
                f"Model generated {len(strings)} strings, exceeding "
                f"max_strings={self.config.max_strings}."
            )

        rendered_image = None
        rendered_image_path = None

        if self.config.save_rendered_image:
            rendered_image = self.rasterizer.render(strings)

            if not isinstance(rendered_image, torch.Tensor):
                raise TypeError(
                    "rasterizer.render() must return a torch.Tensor."
                )

            expected_height, expected_width = self._image_size()

            if tuple(rendered_image.shape) != (
                expected_height,
                expected_width,
            ):
                raise ValueError(
                    "Rasterizer returned an unexpected image shape: "
                    f"{tuple(rendered_image.shape)}; expected "
                    f"({expected_height}, {expected_width})."
                )

            if output_path is None:
                output_path = (
                    Path(self.config.output_dir) / "prediction.png"
                )

            output_path.parent.mkdir(parents=True, exist_ok=True)

            self.rasterizer.save_image(
                rendered_image.detach().cpu(),
                str(output_path),
            )
            rendered_image_path = str(output_path)

            rendered_image = rendered_image.detach().cpu()

        return {
            "tokens": tokens,
            "strings": strings,
            "rendered_image": rendered_image,
            "rendered_image_path": rendered_image_path,
        }

    def predict(self, image: torch.Tensor) -> dict[str, Any]:
        """Generate a string configuration for one grayscale image."""

        self.model.eval()

        return self._predict(image)

    def predict_from_file(
        self,
        image_path: str | Path,
    ) -> dict[str, Any]:
        """Load a grayscale image from disk and run prediction.

        File-based images are resized to the configured model resolution.
        Standard grayscale pixels (0=black, 255=white) are inverted into
        the project's darkness convention (0=white, 1=dark).
        """

        path = Path(image_path)

        if not path.is_file():
            raise FileNotFoundError(f"Input image does not exist: {path}")

        expected_height, expected_width = self._image_size()

        try:
            with Image.open(path) as source:
                grayscale = ImageOps.grayscale(source)
                grayscale = grayscale.resize(
                    (expected_width, expected_height),
                    Image.Resampling.LANCZOS,
                )

                # Convert conventional grayscale into the project's
                # darkness convention: white=0, dark=1.
                pixels = torch.from_numpy(
                    __import__("numpy").array(
                        grayscale,
                        dtype="uint8",
                        copy=True,
                    )
                ).float()

        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Unable to load grayscale image from {path}: {exc}"
            ) from exc

        image = 1.0 - pixels / 255.0

        output_path = (
            Path(self.config.output_dir)
            / f"{path.stem}_reconstruction.png"
        )

        self.model.eval()

        return self._predict(
            image=image,
            output_path=output_path,
        )
