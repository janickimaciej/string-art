from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from configs.model_config import ModelConfig


class IStringTokenizer(ABC):
    """Converts string configurations to and from endpoint-token sequences."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    def encode(
        self,
        strings: Sequence[tuple[int, int]],
        add_special_tokens: bool = True,
    ) -> list[int]:
        """
        Encode strings as a flat endpoint-token sequence.

        Each tuple contains two endpoint tokens, not pin indices.

        Canonicalize endpoints and sort strings deterministically.
        Optionally prepend BOS and append EOS.
        """

    @abstractmethod
    def decode(
        self,
        tokens: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> list[tuple[int, int]]:
        """
        Decode endpoint tokens into string pairs.

        Validate token IDs and complete endpoint pairs.
        Canonicalize endpoints and sort strings deterministically.
        """


class StringTokenizer(IStringTokenizer):
    """Tokenizer for canonical string-art endpoint configurations.

    Endpoint tokens:
        2 * pin_index + side

    Special tokens:
        BOS = 2 * num_pins
        EOS = BOS + 1

    The tokenizer never maps endpoints to text or uses a language tokenizer.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.num_pins = config.num_pins
        self.num_endpoint_tokens = 2 * config.num_pins
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.vocab_size = config.vocab_size

        self._validate_config()

    def _validate_config(self) -> None:
        if self.num_pins < 3:
            raise ValueError("num_pins must be at least 3.")

        if self.bos_token_id != self.num_endpoint_tokens:
            raise ValueError(
                f"BOS must be {self.num_endpoint_tokens}, "
                f"got {self.bos_token_id}."
            )

        if self.eos_token_id != self.bos_token_id + 1:
            raise ValueError(
                f"EOS must be {self.bos_token_id + 1}, "
                f"got {self.eos_token_id}."
            )

        if self.vocab_size != self.eos_token_id + 1:
            raise ValueError(
                f"Vocabulary size must be {self.eos_token_id + 1}, "
                f"got {self.vocab_size}."
            )

    def _validate_endpoint(self, token: int) -> None:
        if isinstance(token, bool) or not isinstance(token, int):
            raise TypeError(
                f"Endpoint token must be an integer, got {token!r}."
            )

        if not 0 <= token < self.num_endpoint_tokens:
            raise ValueError(
                f"Endpoint token {token} is outside the valid range "
                f"[0, {self.num_endpoint_tokens - 1}]."
            )

    def _canonicalize_pair(
        self,
        endpoint_0: int,
        endpoint_1: int,
    ) -> tuple[int, int]:
        self._validate_endpoint(endpoint_0)
        self._validate_endpoint(endpoint_1)

        # The side is encoded in the low bit. Different sides on the
        # same pin are still a degenerate, pin-local connection.
        pin_0 = endpoint_0 // 2
        pin_1 = endpoint_1 // 2

        if pin_0 == pin_1:
            raise ValueError(
                "Degenerate string: both endpoints belong to pin "
                f"{pin_0} (tokens {endpoint_0}, {endpoint_1})."
            )

        return (
            min(endpoint_0, endpoint_1),
            max(endpoint_0, endpoint_1),
        )

    def encode(
        self,
        strings: Sequence[tuple[int, int]],
        add_special_tokens: bool = True,
    ) -> list[int]:
        """Encode and sort a string configuration.

        Every input tuple contains two endpoint token IDs, not pin indices.
        Canonical endpoint order is numeric token order, followed by
        lexicographic sorting of the complete pairs.
        """
        if len(strings) > self.config.max_strings:
            raise ValueError(
                f"Received {len(strings)} strings, exceeding the "
                f"configured maximum of {self.config.max_strings}."
            )

        canonical_pairs: list[tuple[int, int]] = []

        for index, pair in enumerate(strings):
            if len(pair) != 2:
                raise ValueError(
                    f"String {index} must contain exactly two endpoints."
                )

            canonical_pairs.append(
                self._canonicalize_pair(pair[0], pair[1])
            )

        if len(set(canonical_pairs)) != len(canonical_pairs):
            raise ValueError("Duplicate strings are not allowed.")

        canonical_pairs.sort()

        result: list[int] = []

        if add_special_tokens:
            result.append(self.bos_token_id)

        for endpoint_0, endpoint_1 in canonical_pairs:
            result.extend((endpoint_0, endpoint_1))

        if add_special_tokens:
            result.append(self.eos_token_id)

        if len(result) > self.config.max_sequence_length:
            raise ValueError(
                f"Encoded sequence length {len(result)} exceeds the "
                f"configured maximum of "
                f"{self.config.max_sequence_length}."
            )

        return result

    def decode(
        self,
        tokens: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> list[tuple[int, int]]:
        """Decode a token sequence into canonical endpoint pairs.

        When skip_special_tokens=True, a properly framed sequence may
        contain BOS and EOS, and they are removed before decoding.

        When skip_special_tokens=False, special tokens are rejected because
        the returned representation contains endpoint pairs only.
        """
        values = list(tokens)

        for token in values:
            if isinstance(token, bool) or not isinstance(token, int):
                raise TypeError(
                    f"Token IDs must be integers, got {token!r}."
                )

            if not 0 <= token < self.vocab_size:
                raise ValueError(
                    f"Token ID {token} is outside the vocabulary "
                    f"[0, {self.vocab_size - 1}]."
                )

        if skip_special_tokens:
            if (
                len(values) < 2
                or values[0] != self.bos_token_id
                or values[-1] != self.eos_token_id
            ):
                raise ValueError(
                    "Expected a complete sequence framed by BOS and EOS."
                )

            values = values[1:-1]
        else:
            if any(
                token >= self.num_endpoint_tokens for token in values
            ):
                raise ValueError(
                    "Special tokens cannot be decoded as endpoints when "
                    "skip_special_tokens=False."
                )

        if len(values) % 2 != 0:
            raise ValueError(
                "Incomplete string configuration: endpoint token count "
                "must be even."
            )

        if len(values) // 2 > self.config.max_strings:
            raise ValueError(
                "Decoded string count exceeds max_strings."
            )

        pairs: list[tuple[int, int]] = []

        for index in range(0, len(values), 2):
            pairs.append(
                self._canonicalize_pair(
                    values[index],
                    values[index + 1],
                )
            )

        if len(set(pairs)) != len(pairs):
            raise ValueError("Decoded sequence contains duplicate strings.")

        pairs.sort()
        return pairs
