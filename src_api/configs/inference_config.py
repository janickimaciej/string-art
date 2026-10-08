from dataclasses import dataclass


@dataclass
class InferenceConfig:
    """Configuration for generating string-art solutions."""

    # Input / output
    input_path: str
    output_path: str

    # Generation
    max_strings: int
    max_sequence_length: int
    temperature: float

    # Sampling
    do_sample: bool
    top_k: int | None
    top_p: float | None

    # Runtime
    device: str
