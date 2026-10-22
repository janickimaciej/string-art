from dataclasses import asdict, dataclass
import json


@dataclass
class ModelConfig:
    # Image dimensions (model input)
    image_height: int = 512
    image_width: int = 512

    # Circular frame geometry, in image pixels
    num_pins: int = 256
    frame_radius: float = 256.0
    pin_diameter: float = 3.0

    # High-resolution rasterization
    supersampling_factor: int = 8

    # Maximum number of strings
    max_strings: int = 2000

    # Transformer
    hidden_dim: int = 256
    num_heads: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 4
    dropout: float = 0.1

    @property
    def bos_token_id(self) -> int:
        return self.num_pins * 2

    @property
    def eos_token_id(self) -> int:
        return self.bos_token_id + 1

    @property
    def vocab_size(self) -> int:
        return self.eos_token_id + 1

    @property
    def max_sequence_length(self) -> int:
        return 2 * self.max_strings + 2

    def __post_init__(self):
        assert self.num_pins >= 3
        assert self.hidden_dim % self.num_heads == 0
        assert self.max_strings >= 1
        assert self.supersampling_factor >= 1
        assert self.frame_radius > 0
        assert self.pin_diameter > 0

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
            f.write("\n")

    @classmethod
    def from_json(cls, path: str):
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))
