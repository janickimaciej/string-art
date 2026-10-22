from dataclasses import asdict, dataclass
import json


@dataclass
class DataGenerationConfig:
    # Output
    output_dir: str = "artifacts/generated_data/dataset"

    # Dataset sizes
    train_samples: int = 8000
    validation_samples: int = 1000
    test_samples: int = 1000

    # Random string configurations
    min_strings: int = 1
    max_strings: int = 2000
    avg_strings: int = 1000
    stddev_strings: int = 500

    # Reproducibility
    seed: int = 42

    # Generation performance
    samples_per_shard: int = 1000

    def __post_init__(self):
        assert self.train_samples > 0
        assert self.validation_samples > 0
        assert self.test_samples > 0
        assert self.min_strings >= 1
        assert self.max_strings >= self.min_strings
        assert self.samples_per_shard > 0

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
            f.write("\n")

    @classmethod
    def from_json(cls, path: str):
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))
