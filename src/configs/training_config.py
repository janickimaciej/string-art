from dataclasses import asdict, dataclass
import json


@dataclass
class TrainingConfig:
    # Dataset
    dataset_dir: str = "data/generated"

    # Training
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.01

    # Data loading
    num_workers: int = 4

    # Checkpoints and logging
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 100
    seed: int = 42

    def __post_init__(self):
        assert self.batch_size > 0
        assert self.num_epochs > 0
        assert self.learning_rate > 0
        assert self.weight_decay >= 0
        assert self.num_workers >= 0

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
            f.write("\n")

    @classmethod
    def from_json(cls, path: str):
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))
