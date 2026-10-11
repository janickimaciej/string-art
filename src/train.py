from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow execution with: python scripts/training.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_config import ModelConfig
from configs.training_config import TrainingConfig
from data.dataset_loader import DatasetLoader
from data.collator import Collator
from models.encoder_decoder import EncoderDecoder
from training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the string-art Transformer using a previously "
            "generated and persisted image-token dataset."
        )
    )

    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Path to the model configuration JSON file.",
    )

    parser.add_argument(
        "--training-config",
        type=Path,
        required=True,
        help="Path to the training configuration JSON file.",
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help=(
            "Override the dataset directory in TrainingConfig. "
            "Defaults to training_config.dataset_dir."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint from which to resume training.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: auto, cpu, cuda, or cuda:0.",
    )

    parser.add_argument(
        "--max-positions",
        type=int,
        default=1024,
        help="GPT-2 decoder context length.",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Initialize random generators for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    device = torch.device(device_arg)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but CUDA is not available."
        )

    return device


def validate_dataset_directory(dataset_dir: Path) -> None:
    """Fail early if persisted training or validation data is missing."""
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {dataset_dir}. "
            "Generate the dataset before training."
        )

    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Dataset metadata not found: {metadata_path}"
        )

    for split in ("train", "validation"):
        split_dir = dataset_dir / split

        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Required dataset split does not exist: {split_dir}"
            )


def main() -> None:
    args = parse_args()

    if not args.model_config.is_file():
        raise FileNotFoundError(
            f"Model configuration not found: {args.model_config}"
        )

    if not args.training_config.is_file():
        raise FileNotFoundError(
            f"Training configuration not found: {args.training_config}"
        )

    if args.max_positions < 4:
        raise ValueError("--max-positions must be at least 4.")

    # Load the explicitly provided configurations.
    model_config = ModelConfig.from_json(
        str(args.model_config)
    )
    training_config = TrainingConfig.from_json(
        str(args.training_config)
    )

    dataset_dir = (
        args.dataset_dir
        if args.dataset_dir is not None
        else Path(training_config.dataset_dir)
    )

    validate_dataset_directory(dataset_dir)

    set_seed(training_config.seed)
    device = select_device(args.device)

    print(f"Device: {device}")
    print(f"Dataset: {dataset_dir}")
    print(f"Model configuration: {args.model_config}")
    print(f"Training configuration: {args.training_config}")

    # These loaders read persisted examples and validate their metadata.
    # They must not generate examples or render images.
    train_dataset = DatasetLoader(
        dataset_dir=dataset_dir,
        split="train",
        model_config=model_config,
    )

    validation_dataset = DatasetLoader(
        dataset_dir=dataset_dir,
        split="validation",
        model_config=model_config,
    )

    if len(train_dataset) == 0:
        raise ValueError("The training dataset is empty.")

    if len(validation_dataset) == 0:
        raise ValueError("The validation dataset is empty.")

    collator = Collator(model_config)

    loader_kwargs = {
        "batch_size": training_config.batch_size,
        "num_workers": training_config.num_workers,
        "collate_fn": collator,
        "pin_memory": device.type == "cuda",
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )

    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    model = EncoderDecoder(
        config=model_config,
        max_positions=args.max_positions,
    )

    trainer = Trainer(
        model=model,
        model_config=model_config,
        training_config=training_config,
        device=device,
    )

    if args.checkpoint is not None:
        trainer.load_checkpoint(args.checkpoint)

    trainer.train(
        train_loader=train_loader,
        validation_loader=validation_loader,
    )

    print("Training completed.")


if __name__ == "__main__":
    main()
