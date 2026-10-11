from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configs.model_config import ModelConfig
from configs.training_config import TrainingConfig
from models.IEncoderDecoder import IEncoderDecoder


class ITrainer(ABC):
    """Trains the model using supervised learning."""

    def __init__(
        self,
        model: IEncoderDecoder,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.model_config = model_config
        self.training_config = training_config
        self.device = device

    @abstractmethod
    def train(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
    ) -> None:
        """
        Train and validate the model.

        Requirements:
            - Use teacher forcing and token-level cross-entropy.
            - Ignore padded labels (-100).
            - Save checkpoints.
            - Track training and validation loss.
            - Never generate or rasterize training examples.
        """
        raise NotImplementedError

    @abstractmethod
    def train_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> float:
        """Perform one optimization step and return the loss."""
        raise NotImplementedError

    @abstractmethod
    def validate(
        self,
        validation_loader: DataLoader,
    ) -> float:
        """Evaluate mean token-level loss without updating parameters."""
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, path: str | Path) -> None:
        """Save model, optimizer, and training progress."""
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, path: str | Path) -> None:
        """Restore model and training state to resume training."""
        raise NotImplementedError
