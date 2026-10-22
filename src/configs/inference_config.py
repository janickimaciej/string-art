from dataclasses import asdict, dataclass
import json


@dataclass
class InferenceConfig:
    checkpoint_path: str = "artifacts/models/model/best.pt"
    device: str = "auto"

    # Generation
    max_strings: int = 256
    temperature: float = 1.0
    greedy: bool = True

    # Output
    output_dir: str = "outputs"
    save_rendered_image: bool = True

    def __post_init__(self):
        assert self.max_strings > 0
        assert self.temperature > 0

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
            f.write("\n")

    @classmethod
    def from_json(cls, path: str):
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))
