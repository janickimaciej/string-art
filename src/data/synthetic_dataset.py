from __future__ import annotations

import copy
import json
import random
import shutil
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

from configs.model_config import ModelConfig
from configs.data_generation_config import DataGenerationConfig
from data.string_generator import IStringGenerator
from rendering.rasterizer import IRasterizer
from custom_tokenizers.string_tokenizer import IStringTokenizer


class ISyntheticDataset(ABC):
    """Generates and persists synthetic image-token datasets."""

    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataGenerationConfig,
        string_generator: IStringGenerator,
        string_tokenizer: IStringTokenizer,
        rasterizer: IRasterizer,
    ) -> None:
        self.model_config = model_config
        self.data_config = data_config
        self.string_generator = string_generator
        self.string_tokenizer = string_tokenizer
        self.rasterizer = rasterizer

    @abstractmethod
    def generate_and_save(self) -> None:
        """
        Generate and save train, validation, and test splits.

        Each example contains:
            image: grayscale image at configured target resolution.
            tokens: integer sequence [BOS, ..., EOS].

        Requirements:
            - Render using the configured Wu rasterizer.
            - Save image-token pairs in indexed or sharded files.
            - Save metadata describing the generation configuration.
            - Use independent, reproducible random seeds per split.
            - Do not overwrite existing datasets without explicit permission.
        """
        raise NotImplementedError


class SyntheticDataset(ISyntheticDataset):
    """Generate image-token pairs and persist them in compressed shards."""

    FORMAT_VERSION = 1

    SPLIT_SEED_OFFSETS = {
        "train": 0,
        "validation": 1_000_003,
        "test": 2_000_003,
    }

    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataGenerationConfig,
        string_generator: IStringGenerator,
        string_tokenizer: IStringTokenizer,
        rasterizer: IRasterizer,
        allow_overwrite: bool = False,
    ) -> None:
        super().__init__(
            model_config,
            data_config,
            string_generator,
            string_tokenizer,
            rasterizer,
        )
        self.output_dir = Path(data_config.output_dir)
        self.allow_overwrite = allow_overwrite
        self._validate_dependencies()

    def _validate_dependencies(self) -> None:
        config = self.model_config

        if self.string_tokenizer.config != config:
            raise ValueError(
                "Tokenizer configuration does not match ModelConfig."
            )

        if self.rasterizer.config != config:
            raise ValueError(
                "Rasterizer configuration does not match ModelConfig."
            )

        if self.data_config.samples_per_shard < 1:
            raise ValueError("samples_per_shard must be positive.")

        effective_max = min(
            self.data_config.max_strings,
            config.max_strings,
        )

        if not (
            1 <= self.data_config.min_strings <= effective_max
        ):
            raise ValueError("Invalid string-count limits.")

        if self.data_config.stddev_strings < 0:
            raise ValueError("stddev_string must be non-negative.")

        if not (
            self.data_config.min_strings
            <= self.data_config.avg_strings
            <= effective_max
        ):
            raise ValueError(
                "avg_strings must be within the effective string-count "
                "limits."
            )

        if config.bos_token_id != 2 * config.num_pins:
            raise ValueError("Invalid BOS token ID.")

        if config.eos_token_id != config.bos_token_id + 1:
            raise ValueError("Invalid EOS token ID.")

        if config.vocab_size != config.eos_token_id + 1:
            raise ValueError("Invalid vocabulary size.")

    def _make_split_generator(self, seed: int) -> IStringGenerator:
        """Clone the configured generator and assign an independent seed."""
        generator = copy.deepcopy(self.string_generator)
        generator.set_seed(seed)
        return generator

    def _validate_tokens(self, tokens: list[int]) -> None:
        config = self.model_config

        if len(tokens) < 4:
            raise ValueError(
                "Sequence must contain BOS, two endpoint tokens, and EOS."
            )

        if len(tokens) > config.max_sequence_length:
            raise ValueError("Sequence exceeds max_sequence_length.")

        if tokens[0] != config.bos_token_id:
            raise ValueError("Missing BOS token.")

        if tokens[-1] != config.eos_token_id:
            raise ValueError("Missing EOS token.")

        endpoints = tokens[1:-1]

        if len(endpoints) % 2:
            raise ValueError("Endpoint tokens must occur in pairs.")

        if any(
            not isinstance(token, (int, np.integer))
            or not 0 <= token < 2 * config.num_pins
            for token in endpoints
        ):
            raise ValueError("Invalid endpoint token ID.")

        count = len(endpoints) // 2
        maximum = min(
            self.data_config.max_strings,
            config.max_strings,
        )

        if not self.data_config.min_strings <= count <= maximum:
            raise ValueError(
                f"String count {count} is outside the configured limits."
            )

    @staticmethod
    def _image_to_uint8(image: torch.Tensor) -> np.ndarray:
        if not isinstance(image, torch.Tensor):
            raise TypeError("Rasterizer must return a torch.Tensor.")

        if image.ndim != 2:
            raise ValueError("Expected an image with shape [H, W].")

        image = image.detach().to(device="cpu", dtype=torch.float32)

        if not torch.isfinite(image).all().item():
            raise ValueError("Image contains NaN or infinity.")

        if image.min().item() < -1e-6 or image.max().item() > 1 + 1e-6:
            raise ValueError("Image intensities must be in [0, 1].")

        return (
            image.clamp(0, 1)
            .mul(255)
            .round()
            .to(torch.uint8)
            .numpy()
        )

    def _create_example(
        self,
        generator: IStringGenerator,
    ) -> tuple[np.ndarray, list[int]]:
        strings = generator.generate()

        # Persist fully tokenized sequences, including BOS and EOS.
        tokens = self.string_tokenizer.encode(
            strings,
            add_special_tokens=True,
        )
        self._validate_tokens(tokens)

        decoded_strings = self.string_tokenizer.decode(
            tokens,
            skip_special_tokens=True,
        )

        canonical_strings = sorted(
            (min(a, b), max(a, b)) for a, b in strings
        )

        if decoded_strings != canonical_strings:
            raise ValueError("Tokenizer round-trip validation failed.")

        image = self.rasterizer.render(strings)
        image_array = self._image_to_uint8(image)

        expected_shape = (
            self.model_config.image_height,
            self.model_config.image_width,
        )

        if image_array.shape != expected_shape:
            raise ValueError(
                f"Expected image shape {expected_shape}, "
                f"received {image_array.shape}."
            )

        # Verify that the stored token sequence describes the image.
        reconstructed = self.rasterizer.render(decoded_strings)
        reconstructed_array = self._image_to_uint8(reconstructed)

        if not np.array_equal(image_array, reconstructed_array):
            raise ValueError(
                "Rendered image does not match the decoded token sequence."
            )

        return image_array, tokens

    def _save_shard(
        self,
        split_dir: Path,
        shard_index: int,
        examples: list[tuple[np.ndarray, list[int]]],
    ) -> None:
        if not examples:
            return

        images = np.stack([image for image, _ in examples]).astype(
            np.uint8,
            copy=False,
        )

        lengths = np.asarray(
            [len(tokens) for _, tokens in examples],
            dtype=np.int32,
        )

        max_length = int(lengths.max())

        # -1 is a storage-only padding value, not a vocabulary token.
        tokens_array = np.full(
            (len(examples), max_length),
            -1,
            dtype=np.int32,
        )

        for index, (_, tokens) in enumerate(examples):
            tokens_array[index, :len(tokens)] = tokens

        path = split_dir / f"shard_{shard_index:05d}.npz"

        np.savez_compressed(
            path,
            images=images,
            tokens=tokens_array,
            lengths=lengths,
        )

    def _generate_split(
        self,
        split_name: str,
        sample_count: int,
        seed: int,
        root: Path,
    ) -> dict:
        split_dir = root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        generator = self._make_split_generator(seed)
        shard_size = self.data_config.samples_per_shard

        examples = []
        shard_index = 0

        for example_index in range(sample_count):
            try:
                examples.append(self._create_example(generator))
            except Exception as exc:
                raise RuntimeError(
                    f"Failed generating {split_name} example "
                    f"{example_index}: {exc}"
                ) from exc

            if len(examples) == shard_size:
                self._save_shard(split_dir, shard_index, examples)
                examples.clear()
                shard_index += 1

        if examples:
            self._save_shard(split_dir, shard_index, examples)
            shard_index += 1

        return {
            "num_examples": sample_count,
            "num_shards": shard_index,
            "seed": seed,
        }

    def _build_metadata(self, split_metadata: dict) -> dict:
        config = self.model_config
        data_config = self.data_config

        return {
            "format_version": self.FORMAT_VERSION,
            "storage_format": "compressed_npz_shards",
            "image": {
                "height": config.image_height,
                "width": config.image_width,
                "dtype": "uint8",
                "intensity_convention": (
                    "0=white background, 255=maximum darkness"
                ),
                "training_normalization": "float32 / 255.0",
            },
            "geometry": {
                "num_pins": config.num_pins,
                "frame_center": [
                    config.image_width / 2.0,
                    config.image_height / 2.0,
                ],
                "frame_radius": config.frame_radius,
                "pin_diameter": config.pin_diameter,
            },
            "rasterization": {
                "algorithm": "Xiaolin Wu",
                "supersampling_factor": config.supersampling_factor,
                "high_resolution_string_width": 1.0,
                "intensity_convention": "additive darkness, clamped to [0, 1]",
            },
            "tokens": {
                "endpoint_token_min": 0,
                "endpoint_token_max": 2 * config.num_pins - 1,
                "bos_token_id": config.bos_token_id,
                "eos_token_id": config.eos_token_id,
                "vocab_size": config.vocab_size,
                "max_strings": min(
                    data_config.max_strings,
                    config.max_strings,
                ),
                "max_sequence_length": config.max_sequence_length,
            },
            "string_sampling": {
                "min_strings": data_config.min_strings,
                "max_strings": min(
                    data_config.max_strings,
                    config.max_strings,
                ),
                "avg_strings": data_config.avg_strings,
                "stddev_string": data_config.stddev_strings,
                "count_distribution": "clipped normal",
                "duplicate_strings": False,
                "self_loops": False,
                "same_pin_connections": False,
            },
            "sharding": {
                "samples_per_shard": data_config.samples_per_shard,
                "arrays": ["images", "tokens", "lengths"],
                "token_dtype": "int32",
                "padding_value": -1,
            },
            "splits": split_metadata,
            "generation_config": {
                "model_config": vars(config),
                "data_generation_config": vars(data_config),
            },
        }

    def generate_and_save(self) -> None:
        output_dir = self.output_dir
        output_dir.parent.mkdir(parents=True, exist_ok=True)

        if output_dir.exists() and not self.allow_overwrite:
            raise FileExistsError(
                f"Dataset already exists: {output_dir}. "
                "Choose a new output directory or set allow_overwrite=True."
            )

        temp_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.tmp-",
                dir=str(output_dir.parent),
            )
        )

        split_sizes = {
            "train": self.data_config.train_samples,
            "validation": self.data_config.validation_samples,
            "test": self.data_config.test_samples,
        }

        split_metadata = {}

        try:
            for split_name, sample_count in split_sizes.items():
                seed = (
                    self.data_config.seed
                    + self.SPLIT_SEED_OFFSETS[split_name]
                )

                split_metadata[split_name] = self._generate_split(
                    split_name,
                    sample_count,
                    seed,
                    temp_dir,
                )

            metadata = self._build_metadata(split_metadata)

            with (temp_dir / "metadata.json").open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(metadata, file, indent=2)
                file.write("\n")

            if output_dir.exists():
                backup_dir = output_dir.with_name(
                    output_dir.name + ".backup"
                )

                if backup_dir.exists():
                    raise FileExistsError(
                        f"Backup directory already exists: {backup_dir}"
                    )

                os.replace(output_dir, backup_dir)

                try:
                    os.replace(temp_dir, output_dir)
                except Exception:
                    os.replace(backup_dir, output_dir)
                    raise
                else:
                    shutil.rmtree(backup_dir)
            else:
                os.replace(temp_dir, output_dir)

        except Exception:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        print(f"Dataset saved to {output_dir}")

        for split, info in split_metadata.items():
            print(
                f"{split}: {info['num_examples']} examples, "
                f"{info['num_shards']} shards, seed={info['seed']}"
            )
