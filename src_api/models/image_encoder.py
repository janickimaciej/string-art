from torch import Tensor, nn


class ImageEncoder(nn.Module):
    """Transformer encoder operating on tokenized image features."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1,
        max_num_tokens: int = 256,
    ) -> None:
        """
        Initialize the image Transformer encoder.

        Args:
            hidden_dim: Dimension of the image token embeddings.
            num_layers: Number of Transformer encoder layers.
            num_heads: Number of attention heads per layer.
            feedforward_dim: Dimension of the Transformer feed-forward layers.
            dropout: Dropout probability used by the Transformer.
            max_num_tokens: Maximum number of image tokens supported.
        """
        ...

    def forward(
        self,
        image_tokens: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Encode image tokens into contextualized visual representations.

        Args:
            image_tokens: Image tokens with shape
                [batch, num_tokens, hidden_dim].
            attention_mask: Optional mask indicating valid image tokens.

        Returns:
            Encoder hidden states with shape
            [batch, num_tokens, hidden_dim].
        """
        ...
