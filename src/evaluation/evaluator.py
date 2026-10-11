from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.utils.data import DataLoader

from models.IEncoderDecoder import IEncoderDecoder
from tokenizers.IStringTokenizer import IStringTokenizer
from rendering.IRasterizer import IRasterizer


class IEvaluator(ABC):
    """Evaluates generated strings and rendered images."""

    def __init__(
        self,
        model: IEncoderDecoder,
        tokenizer: IStringTokenizer,
        rasterizer: IRasterizer,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.rasterizer = rasterizer
        self.device = device

    @abstractmethod
    def evaluate(
        self,
        data_loader: DataLoader,
    ) -> dict[str, float]:
        """
        Evaluate a saved validation or test dataset.

        Return metrics such as:
            - token_loss
            - token_accuracy
            - exact_string_pair_accuracy
            - mean_image_mse

        Image metrics must be calculated by generating sequences,
        rendering the predicted strings, and comparing the rendered
        images with the corresponding target images.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_example(
        self,
        image: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> dict[str, Any]:
        """
        Evaluate one example.

        Return metrics and optionally the generated tokens,
        decoded strings, and rendered reconstruction.
        """
        raise NotImplementedError
