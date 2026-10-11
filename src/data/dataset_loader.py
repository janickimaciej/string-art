from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from configs.model_config import ModelConfig


class IDatasetLoader(Dataset, ABC):
    """Reads previously generated image-token pairs from disk."""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        model_config: ModelConfig,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.model_config = model_config

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of saved examples in this split."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Load one saved example.

        Returns:
            {
                "image": Tensor[C, H, W],
                "tokens": Tensor[T],
            }

        Requirements:
            - Read only persisted dataset files.
            - Validate dataset metadata and configuration compatibility.
            - Preserve BOS and EOS.
            - Do not pad sequences here; padding belongs in collator.py.
        """
        raise NotImplementedError


class DatasetLoader(IDatasetLoader):
    """Disk-backed loader for compressed NPZ dataset shards.

    Expected layout:
        dataset_dir/
            metadata.json
            train/shard_00000.npz
            validation/shard_00000.npz
            test/shard_00000.npz

    Each shard contains:
        images: uint8 [B, H, W]
        tokens: int32 [B, T], padded with -1
        lengths: int32 [B]

    Returned examples contain:
        image: float32 Tensor [1, H, W], normalized to [0, 1]
        tokens: int64 Tensor [T], including BOS and EOS

    Padding is removed using lengths. Batch padding is the responsibility
    of the collate function.
    """

    SUPPORTED_FORMAT_VERSION = 1
    VALID_SPLITS = {"train", "validation", "test"}

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        model_config: ModelConfig,
        shard_cache_size: int = 2,
    ) -> None:
        super().__init__(dataset_dir, split, model_config)

        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"Invalid split {split!r}. "
                f"Expected one of {sorted(self.VALID_SPLITS)}."
            )

        if shard_cache_size < 1:
            raise ValueError("shard_cache_size must be at least 1.")

        self.shard_cache_size = shard_cache_size
        self._shard_cache: OrderedDict[
            int, dict[str, np.ndarray]
        ] = OrderedDict()

        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {self.dataset_dir}"
            )

        self.metadata_path = self.dataset_dir / "metadata.json"

        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                f"Dataset metadata not found: {self.metadata_path}"
            )

        with self.metadata_path.open("r", encoding="utf-8") as file:
            self.metadata = json.load(file)

        self._validate_metadata()

        self.split_dir = self.dataset_dir / split

        if not self.split_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset split directory does not exist: {self.split_dir}"
            )

        # Ignore unrelated files; accept only indexed shard filenames.
        shard_pattern = re.compile(r"^shard_(\d+)\.npz$")
        indexed_shards = []

        for path in self.split_dir.iterdir():
            match = shard_pattern.match(path.name)
            if match:
                indexed_shards.append((int(match.group(1)), path))

        indexed_shards.sort(key=lambda item: item[0])

        if not indexed_shards:
            raise FileNotFoundError(
                f"No indexed NPZ shards found in {self.split_dir}."
            )

        indices = [index for index, _ in indexed_shards]
        if indices != list(range(len(indices))):
            raise ValueError(
                f"Shard indices in {self.split_dir} must be contiguous "
                "and start at zero."
            )

        self.shard_paths = [path for _, path in indexed_shards]
        self.shard_lengths: list[int] = []
        self.cumulative_lengths: list[int] = []

        total = 0

        for path in self.shard_paths:
            try:
                with np.load(path, allow_pickle=False) as shard:
                    self._validate_shard_keys(shard, path)
                    lengths = shard["lengths"]

                    if lengths.ndim != 1 or lengths.dtype != np.int32:
                        raise ValueError(
                            f"{path}: lengths must be a 1D int32 array."
                        )

                    count = len(lengths)

                    if count == 0:
                        raise ValueError(f"{path}: empty shards are invalid.")

                    if np.any(lengths < 4):
                        raise ValueError(
                            f"{path}: sequences must contain BOS, at "
                            "least two endpoint tokens, and EOS."
                        )

                    if np.any(
                        lengths > model_config.max_sequence_length
                    ):
                        raise ValueError(
                            f"{path}: a sequence exceeds "
                            "max_sequence_length."
                        )

                    tokens = shard["tokens"]

                    if tokens.ndim != 2 or tokens.dtype != np.int32:
                        raise ValueError(
                            f"{path}: tokens must be a 2D int32 array."
                        )

                    if tokens.shape[0] != count:
                        raise ValueError(
                            f"{path}: token and length counts differ."
                        )

                    if tokens.shape[1] < int(lengths.max()):
                        raise ValueError(
                            f"{path}: token array is shorter than a "
                            "recorded sequence length."
                        )

            except (OSError, ValueError, KeyError) as exc:
                raise ValueError(
                    f"Invalid dataset shard {path}: {exc}"
                ) from exc

            self.shard_lengths.append(count)
            total += count
            self.cumulative_lengths.append(total)

        self.num_examples = total

        expected_count = self.metadata["splits"][split]["num_examples"]
        if self.num_examples != expected_count:
            raise ValueError(
                f"Split {split!r} contains {self.num_examples} examples, "
                f"but metadata declares {expected_count}."
            )

    def _validate_metadata(self) -> None:
        """Fail early when stored data and the active model disagree."""
        metadata = self.metadata
        config = self.model_config

        version = metadata.get("format_version")
        if version != self.SUPPORTED_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported dataset format version {version!r}; "
                f"expected {self.SUPPORTED_FORMAT_VERSION}."
            )

        if metadata.get("storage_format") != "compressed_npz_shards":
            raise ValueError("Unsupported dataset storage format.")

        image = metadata.get("image", {})
        expected_image = {
            "height": config.image_height,
            "width": config.image_width,
            "dtype": "uint8",
            "intensity_convention": (
                "0=white background, 255=maximum darkness"
            ),
        }

        for key, expected in expected_image.items():
            actual = image.get(key)
            if actual != expected:
                raise ValueError(
                    f"Incompatible dataset image setting {key!r}: "
                    f"dataset={actual!r}, model={expected!r}."
                )

        geometry = metadata.get("geometry", {})
        expected_geometry = {
            "num_pins": config.num_pins,
            "frame_radius": config.frame_radius,
            "pin_diameter": config.pin_diameter,
        }

        for key, expected in expected_geometry.items():
            actual = geometry.get(key)
            if actual != expected:
                raise ValueError(
                    f"Incompatible geometry setting {key!r}: "
                    f"dataset={actual!r}, model={expected!r}."
                )

        expected_center = [
            config.image_width / 2.0,
            config.image_height / 2.0,
        ]

        if geometry.get("frame_center") != expected_center:
            raise ValueError(
                "Dataset frame center does not match ModelConfig."
            )

        rasterization = metadata.get("rasterization", {})
        if (
            rasterization.get("supersampling_factor")
            != config.supersampling_factor
        ):
            raise ValueError(
                "Dataset supersampling factor does not match ModelConfig."
            )

        tokens = metadata.get("tokens", {})
        expected_tokens = {
            "endpoint_token_min": 0,
            "endpoint_token_max": 2 * config.num_pins - 1,
            "bos_token_id": config.bos_token_id,
            "eos_token_id": config.eos_token_id,
            "vocab_size": config.vocab_size,
            "max_sequence_length": config.max_sequence_length,
        }

        for key, expected in expected_tokens.items():
            actual = tokens.get(key)
            if actual != expected:
                raise ValueError(
                    f"Incompatible token setting {key!r}: "
                    f"dataset={actual!r}, model={expected!r}."
                )

        if metadata.get("padding_token_id") not in (None, -1):
            raise ValueError("Unsupported metadata padding token ID.")

        sharding = metadata.get("sharding", {})
        if sharding.get("padding_value") != -1:
            raise ValueError("Expected -1 as the storage padding value.")

    @staticmethod
    def _validate_shard_keys(shard: Any, path: Path) -> None:
        required = {"images", "tokens", "lengths"}
        missing = required.difference(shard.files)

        if missing:
            raise ValueError(
                f"{path}: missing required arrays {sorted(missing)}."
            )

    def __len__(self) -> int:
        return self.num_examples

    def _locate_index(self, index: int) -> tuple[int, int]:
        """Map a global dataset index to (shard index, local index)."""
        if index < 0:
            index += self.num_examples

        if not 0 <= index < self.num_examples:
            raise IndexError(
                f"Dataset index {index} out of range "
                f"for {self.num_examples} examples."
            )

        # Binary search over cumulative shard sizes.
        left, right = 0, len(self.cumulative_lengths)

        while left < right:
            middle = (left + right) // 2
            if index < self.cumulative_lengths[middle]:
                right = middle
            else:
                left = middle + 1

        shard_index = left
        previous_end = (
            self.cumulative_lengths[shard_index - 1]
            if shard_index > 0
            else 0
        )

        return shard_index, index - previous_end

    def _get_shard(self, shard_index: int) -> dict[str, np.ndarray]:
        """Load a shard into a bounded LRU cache."""
        if shard_index in self._shard_cache:
            self._shard_cache.move_to_end(shard_index)
            return self._shard_cache[shard_index]

        path = self.shard_paths[shard_index]

        with np.load(path, allow_pickle=False) as archive:
            arrays = {
                "images": archive["images"],
                "tokens": archive["tokens"],
                "lengths": archive["lengths"],
            }

        images = arrays["images"]
        tokens = arrays["tokens"]
        lengths = arrays["lengths"]

        expected_image_shape = (
            self.shard_lengths[shard_index],
            self.model_config.image_height,
            self.model_config.image_width,
        )

        if images.shape != expected_image_shape:
            raise ValueError(
                f"{path}: image shape {images.shape}, expected "
                f"{expected_image_shape}."
            )

        if images.dtype != np.uint8:
            raise ValueError(f"{path}: images must have dtype uint8.")

        if tokens.dtype != np.int32 or tokens.ndim != 2:
            raise ValueError(f"{path}: tokens must be a 2D int32 array.")

        if tokens.shape[0] != len(images) or len(lengths) != len(images):
            raise ValueError(f"{path}: inconsistent array lengths.")

        if lengths.dtype != np.int32:
            raise ValueError(f"{path}: lengths must have dtype int32.")

        self._shard_cache[shard_index] = arrays
        self._shard_cache.move_to_end(shard_index)

        while len(self._shard_cache) > self.shard_cache_size:
            self._shard_cache.popitem(last=False)

        return arrays

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard_index, local_index = self._locate_index(index)
        shard = self._get_shard(shard_index)

        image_array = shard["images"][local_index]
        token_array = shard["tokens"][local_index]
        length = int(shard["lengths"][local_index])

        if not 4 <= length <= self.model_config.max_sequence_length:
            raise ValueError(
                f"Invalid sequence length {length} at dataset index {index}."
            )

        # Remove storage padding. Preserve both BOS and EOS.
        token_array = token_array[:length]

        config = self.model_config

        if token_array[0] != config.bos_token_id:
            raise ValueError(f"Example {index} is missing BOS.")

        if token_array[-1] != config.eos_token_id:
            raise ValueError(f"Example {index} is missing EOS.")

        endpoints = token_array[1:-1]

        if len(endpoints) % 2:
            raise ValueError(
                f"Example {index} has an incomplete endpoint pair."
            )

        if np.any(endpoints < 0) or np.any(
            endpoints >= 2 * config.num_pins
        ):
            raise ValueError(
                f"Example {index} contains invalid endpoint token IDs."
            )

        # Images use the convention 0=white, 1=maximum darkness.
        image = torch.from_numpy(image_array.copy())
        image = image.to(torch.float32).div_(255.0).unsqueeze(0)

        tokens = torch.from_numpy(token_array.copy()).to(torch.long)

        return {
            "image": image,
            "tokens": tokens,
        }
