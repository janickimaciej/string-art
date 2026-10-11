import torch
import torch.nn.functional as F


def token_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute token-level cross-entropy, ignoring padded labels.

    Args:
        logits: [B, T, vocab_size]
        labels: [B, T], with -100 at padding positions.

    Returns:
        Scalar mean cross-entropy loss.
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )
