from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from configs.model_config import ModelConfig
from configs.training_config import TrainingConfig
from models.encoder_decoder import IEncoderDecoder


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


class Trainer(ITrainer):
    """Supervised trainer for the image-to-string Transformer."""

    def __init__(
        self,
        model: IEncoderDecoder,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        device: torch.device,
    ) -> None:
        super().__init__(
            model,
            model_config,
            training_config,
            device,
        )

        self.model.to(self.device)

        self.epochs = int(
            getattr(training_config, "epochs", 10)
        )
        learning_rate = float(
            getattr(training_config, "learning_rate", 3e-4)
        )
        weight_decay = float(
            getattr(training_config, "weight_decay", 0.01)
        )
        self.gradient_clip_norm = float(
            getattr(training_config, "gradient_clip_norm", 1.0)
        )

        self.model_dir = Path(
            getattr(training_config, "model_dir", "checkpoints")
        )

        self.save_every_epochs = int(
            getattr(training_config, "save_every_epochs", 1)
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=float(
                getattr(training_config, "scheduler_factor", 0.5)
            ),
            patience=int(
                getattr(training_config, "scheduler_patience", 2)
            ),
        )

        self.epoch = 0
        self.global_step = 0
        self.best_validation_loss = float("inf")
        self.history: list[dict[str, float]] = []

    def _prepare_batch(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Consume batches already shifted by the collator."""

        required = {
            "images",
            "decoder_input_ids",
            "labels",
            "attention_mask",
        }

        missing = required - batch.keys()
        if missing:
            raise KeyError(
                f"Batch is missing required keys: {sorted(missing)}"
            )

        images = batch["images"].to(
            self.device,
            dtype=torch.float32,
        )
        decoder_input_ids = batch["decoder_input_ids"].to(
            self.device,
            dtype=torch.long,
        )
        labels = batch["labels"].to(
            self.device,
            dtype=torch.long,
        )
        attention_mask = batch["attention_mask"].to(
            self.device,
            dtype=torch.long,
        )

        if decoder_input_ids.shape != labels.shape:
            raise ValueError(
                "decoder_input_ids and labels must have the same shape."
            )

        if attention_mask.shape != decoder_input_ids.shape:
            raise ValueError(
                "attention_mask must match decoder_input_ids."
            )

        if images.shape[0] != decoder_input_ids.shape[0]:
            raise ValueError(
                "Image and decoder batch sizes must match."
            )

        return images, decoder_input_ids, attention_mask, labels

    def train_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> float:
        """Run one supervised optimization step."""

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        (
            images,
            decoder_input_ids,
            attention_mask,
            labels,
        ) = self._prepare_batch(batch)

        logits = self.model(
            images=images,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
        )

        loss = F.cross_entropy(
            logits.reshape(-1, self.model_config.vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at step {self.global_step}."
            )

        loss.backward()

        if self.gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip_norm,
            )

        self.optimizer.step()
        self.global_step += 1

        return loss.detach().item()

    @torch.no_grad()
    def validate(
        self,
        validation_loader: DataLoader,
    ) -> float:
        """Compute mean cross-entropy over valid target tokens."""

        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        for batch in validation_loader:
            (
                images,
                decoder_input_ids,
                attention_mask,
                labels,
            ) = self._prepare_batch(batch)

            logits = self.model(
                images=images,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
            )

            loss_sum = F.cross_entropy(
                logits.reshape(-1, self.model_config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )

            total_loss += loss_sum.item()
            total_tokens += labels.ne(-100).sum().item()

        if total_tokens == 0:
            raise ValueError(
                "Validation dataset contains no valid target tokens."
            )

        return total_loss / total_tokens

    def train(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
    ) -> None:
        """Train for the configured number of epochs."""

        if len(train_loader) == 0:
            raise ValueError("Training DataLoader is empty.")

        if len(validation_loader) == 0:
            raise ValueError("Validation DataLoader is empty.")

        self.model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for epoch_index in range(self.epoch, self.epochs):
            epoch_number = epoch_index + 1
            total_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                loss = self.train_step(batch)
                total_loss += loss
                num_batches += 1

            train_loss = total_loss / num_batches
            validation_loss = self.validate(validation_loader)

            self.scheduler.step(validation_loss)
            self.epoch = epoch_number

            metrics = {
                "epoch": float(epoch_number),
                "global_step": float(self.global_step),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
            self.history.append(metrics)

            print(
                f"Epoch {epoch_number}/{self.epochs} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={validation_loss:.6f}"
            )

            if validation_loss < self.best_validation_loss:
                self.best_validation_loss = validation_loss
                self.save_checkpoint(
                    self.model_dir / "best.pt"
                )

            if epoch_number % self.save_every_epochs == 0:
                self.save_checkpoint(
                    self.model_dir / "last.pt"
                )

        self.save_checkpoint(
            self.model_dir / "last.pt"
        )

    def save_checkpoint(
        self,
        path: str | Path,
    ) -> None:
        """Save model weights and training state."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_validation_loss": self.best_validation_loss,
            "history": self.history,
            "model_config": self.model_config.__dict__.copy(),
        }

        temporary_path = path.with_suffix(path.suffix + ".tmp")

        try:
            torch.save(checkpoint, temporary_path)
            temporary_path.replace(path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def load_checkpoint(
        self,
        path: str | Path,
    ) -> None:
        """Restore model and optimizer state to resume training."""

        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

        try:
            checkpoint = torch.load(
                path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                path,
                map_location=self.device,
            )

        saved_config = checkpoint.get("model_config")

        if saved_config is not None:
            current_config = self.model_config.__dict__

            for key, value in current_config.items():
                if saved_config.get(key) != value:
                    raise ValueError(
                        f"Checkpoint configuration mismatch for {key}: "
                        f"saved={saved_config.get(key)!r}, "
                        f"current={value!r}."
                    )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        self.epoch = int(checkpoint["epoch"])
        self.global_step = int(checkpoint["global_step"])
        self.best_validation_loss = float(
            checkpoint.get(
                "best_validation_loss",
                float("inf"),
            )
        )
        self.history = checkpoint.get("history", [])

        self.model.to(self.device)

        print(
            f"Resumed training from {path}: "
            f"epoch={self.epoch}, step={self.global_step}"
        )
