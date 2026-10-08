from torch import Tensor, nn

from .decoder import Decoder
from .image_encoder import ImageEncoder


class EncoderDecoder(nn.Module):
    """Complete image-to-string Transformer model."""

    def __init__(
        self,
        image_encoder: ImageEncoder,
        decoder: Decoder,
    ) -> None:
        """
        Initialize the encoder-decoder model.

        Args:
            image_encoder: Transformer encoder that processes image tokens.
            decoder: GPT-2-based decoder with cross-attention.
        """
        ...

    def encode(
        self,
        image_tokens: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Encode tokenized images into encoder hidden states.

        Args:
            image_tokens: Image tokens with shape
                [batch, num_image_tokens, hidden_dim].
            attention_mask: Optional mask for image tokens.

        Returns:
            Encoder hidden states.
        """
        ...

    def decode(
        self,
        input_ids: Tensor,
        encoder_hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Predict the next-token distributions conditioned on encoded images.

        Args:
            input_ids: Decoder input IDs with shape
                [batch, sequence_length].
            encoder_hidden_states: Image encoder output.
            attention_mask: Optional decoder attention mask.
            encoder_attention_mask: Optional encoder attention mask.

        Returns:
            Token logits with shape [batch, sequence_length, vocab_size].
        """
        ...

    def forward(
        self,
        image_tokens: Tensor,
        input_ids: Tensor,
        image_attention_mask: Tensor | None = None,
        decoder_attention_mask: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Run the complete encoder-decoder model.

        Args:
            image_tokens: Tokenized images with shape
                [batch, num_image_tokens, hidden_dim].
            input_ids: Decoder input IDs with shape
                [batch, sequence_length].
            image_attention_mask: Optional mask for image tokens.
            decoder_attention_mask: Optional causal/padding mask for decoder.
            encoder_attention_mask: Optional mask applied during
                decoder cross-attention.

        Returns:
            Token logits with shape [batch, sequence_length, vocab_size].
        """
        ...
