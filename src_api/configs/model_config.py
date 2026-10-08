from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the image-to-string encoder-decoder model."""

    # Image / image tokenizer
    image_channels: int
    image_size: int
    image_patch_size: int

    # Image encoder
    encoder_hidden_dim: int
    encoder_num_layers: int
    encoder_num_heads: int
    encoder_feedforward_dim: int
    encoder_dropout: float

    # Decoder
    decoder_hidden_dim: int
    decoder_num_layers: int
    decoder_num_heads: int
    decoder_feedforward_dim: int
    decoder_dropout: float
    decoder_max_sequence_length: int
