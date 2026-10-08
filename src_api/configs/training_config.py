from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for reinforcement-learning training."""

    # Optimization
    learning_rate: float
    weight_decay: float
    num_epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    max_grad_norm: float

    # Reinforcement learning
    rl_algorithm: str
    temperature: float
    reward_scale: float
    baseline_type: str

    # Generation during training
    max_strings: int
    max_sequence_length: int

    # Checkpointing / logging
    checkpoint_interval: int
    log_interval: int
    output_dir: str

    # Reproducibility
    seed: int
