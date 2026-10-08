from torch import Tensor, nn


class Decoder(nn.Module):
    """GPT-2-based autoregressive decoder with encoder-decoder cross-attention."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        max_sequence_length: int,
        feedforward_dim: int | None = None,
        dropout: float = 0.1,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
    ) -> None:
        """
        Initialize the GPT-2 decoder.

        Args:
            vocab_size: Number of tokens in the decoder vocabulary.
            hidden_dim: Dimension of decoder token representations.
            num_layers: Number of GPT-2 decoder layers.
            num_heads: Number of self-attention heads per layer.
            max_sequence_length: Maximum number of generated tokens.
            feedforward_dim: Dimension of the decoder feed-forward layers.
                Uses the GPT-2 default scaling when None.
            dropout: Dropout probability used by the decoder.
            bos_token_id: Token ID used to mark the beginning of generation.
            eos_token_id: Token ID used to mark the end of generation.
        """
        ...

    def forward(
        self,
        input_ids: Tensor,
        encoder_hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> Tensor:
        """
        Generate token logits conditioned on the encoded image.

        Args:
            input_ids: Decoder input token IDs with shape
                [batch, sequence_length].
            encoder_hidden_states: Output of the image encoder with shape
                [batch, num_image_tokens, hidden_dim].
            attention_mask: Optional decoder attention mask.
            encoder_attention_mask: Optional mask for encoder hidden states.
            labels: Optional target token IDs used to calculate the
                autoregressive language-model loss.

        Returns:
            Token logits with shape [batch, sequence_length, vocab_size].
        """
        ...
