from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from models.encoder_decoder import EncoderDecoder
from rendering.geometry import String
from rewards.reward import StringArtReward
from tokenizers.image_tokenizer import ImageTokenizer
from tokenizers.string_tokenizer import StringTokenizer


class Training:
    """Reinforcement-learning training loop for the string-art model."""

    def __init__(
        self,
        model: EncoderDecoder,
        image_tokenizer: ImageTokenizer,
        string_tokenizer: StringTokenizer,
        reward: StringArtReward,
        optimizer: Optimizer,
        device: str,
        max_sequence_length: int,
        temperature: float = 1.0,
        max_strings: int | None = None,
        gradient_clip_norm: float | None = None,
    ) -> None:
        """
        Initialize the RL training pipeline.

        Args:
            model: Image-to-string encoder-decoder model.
            image_tokenizer: Converts images into encoder input tokens.
            string_tokenizer: Converts sampled tokens into strings.
            reward: Computes image-based rewards for generated strings.
            optimizer: Optimizer used to update model parameters.
            device: Device on which training is performed.
            max_sequence_length: Maximum number of tokens generated per sample.
            temperature: Sampling temperature for autoregressive generation.
            max_strings: Optional maximum number of strings generated per image.
            gradient_clip_norm: Optional maximum gradient norm.
        """
        ...

    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int,
    ) -> None:
        """
        Train the model using reinforcement learning.

        Args:
            dataloader: Dataset providing target images.
            num_epochs: Number of passes through the training dataset.
        """
        ...

    def train_step(
        self,
        images: Tensor,
    ) -> dict[str, float]:
        """
        Perform one reinforcement-learning training step.

        Args:
            images: Batch of target images with shape
                [batch, channels, height, width].

        Returns:
            Dictionary containing training metrics such as reward,
            image loss, and policy loss.
        """
        ...

    def generate_rollout(
        self,
        images: Tensor,
    ) -> tuple[list[list[int]], Tensor]:
        """
        Sample token sequences and collect their log probabilities.

        Args:
            images: Batch of target images.

        Returns:
            Tuple containing:
                - sampled token sequences, one per image.
                - summed log probabilities of the sampled sequences.
        """
        ...

    def compute_policy_loss(
        self,
        log_probabilities: Tensor,
        rewards: Tensor,
    ) -> Tensor:
        """
        Compute the policy-gradient loss from sampled trajectories.

        Args:
            log_probabilities: Log probabilities of sampled trajectories.
            rewards: Rewards assigned to each trajectory.

        Returns:
            Scalar policy-gradient loss.
        """
        ...

    def save_checkpoint(
        self,
        path: str,
    ) -> None:
        """
        Save the current training state.

        Args:
            path: Path where the checkpoint should be written.
        """
        ...

    def load_checkpoint(
        self,
        path: str,
    ) -> None:
        """
        Restore a previously saved training state.

        Args:
            path: Path to the checkpoint.
        """
        ...
