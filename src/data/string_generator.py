from abc import ABC, abstractmethod
import random

from configs.model_config import ModelConfig
from configs.data_generation_config import DataGenerationConfig


class IStringGenerator(ABC):
    """Generates random, canonical string configurations."""

    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataGenerationConfig,
    ) -> None:
        self.model_config = model_config
        self.data_config = data_config

    @abstractmethod
    def generate(self) -> list[tuple[int, int]]:
        """
        Generate a random string configuration.

        Returns:
            A list of (endpoint_token_0, endpoint_token_1) pairs.

        Requirements:
            - Each endpoint token is in [0, 511].
            - Canonicalize endpoints and sort strings deterministically.
            - Respect configured string-count limits.
            - Respect duplicate and degenerate-string constraints.
        """
        raise NotImplementedError

    @abstractmethod
    def set_seed(self, seed: int) -> None:
        raise NotImplementedError


class StringGenerator(IStringGenerator):
    """Generate reproducible random string configurations.

    Each string connects attachment points on two distinct pins.
    Duplicate strings are excluded, and endpoints and strings are
    returned in canonical lexicographic order.

    The number of strings follows a clipped normal distribution
    parameterized by avg_strings and stddev_strings.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataGenerationConfig,
    ) -> None:
        super().__init__(model_config, data_config)

        self._rng = random.Random(data_config.seed)

        self._min_strings = data_config.min_strings
        self._max_strings = min(
            data_config.max_strings,
            model_config.max_strings,
        )

        if self._min_strings > self._max_strings:
            raise ValueError(
                "Invalid string-count limits: min_strings "
                f"({self._min_strings}) exceeds the effective maximum "
                f"({self._max_strings})."
            )

        if data_config.stddev_strings < 0:
            raise ValueError(
                "stddev_strings must be non-negative."
            )

        if not (
            self._min_strings
            <= data_config.avg_strings
            <= self._max_strings
        ):
            raise ValueError(
                "avg_strings must lie within the effective string-count "
                f"range [{self._min_strings}, {self._max_strings}]."
            )

        self._num_pins = model_config.num_pins
        self._num_endpoint_tokens = 2 * self._num_pins

        self._validate_token_configuration()

        # Precompute all valid, canonical strings.
        # A valid string connects two different pins, with either
        # attachment side allowed at each endpoint.
        #
        # Each pair is already canonical because the first pin index
        # is smaller than the second pin index.
        self._valid_pairs = tuple(
            (
                2 * pin_a + side_a,
                2 * pin_b + side_b,
            )
            for pin_a in range(self._num_pins)
            for pin_b in range(pin_a + 1, self._num_pins)
            for side_a in range(2)
            for side_b in range(2)
        )

        self._max_unique_strings = len(self._valid_pairs)

        if self._min_strings > self._max_unique_strings:
            raise ValueError(
                "min_strings exceeds the number of unique, "
                f"non-degenerate strings ({self._max_unique_strings})."
            )

        self._max_strings = min(
            self._max_strings,
            self._max_unique_strings,
        )

    def _validate_token_configuration(self) -> None:
        """Ensure the model uses the required endpoint-token mapping."""
        expected_bos = self._num_endpoint_tokens
        expected_eos = expected_bos + 1
        expected_vocab_size = expected_eos + 1

        if (
            self.model_config.bos_token_id != expected_bos
            or self.model_config.eos_token_id != expected_eos
            or self.model_config.vocab_size != expected_vocab_size
        ):
            raise ValueError(
                "Incompatible token configuration: expected endpoint "
                f"tokens [0, {expected_bos - 1}], BOS={expected_bos}, "
                f"EOS={expected_eos}, and vocabulary size "
                f"{expected_vocab_size}."
            )

    def _sample_string_count(self) -> int:
        """Sample a bounded, approximately normal string count."""
        mean = self.data_config.avg_strings
        stddev = self.data_config.stddev_strings

        if stddev == 0:
            count = mean
        else:
            count = round(self._rng.gauss(mean, stddev))

        return max(
            self._min_strings,
            min(int(count), self._max_strings),
        )

    def generate(self) -> list[tuple[int, int]]:
        """Generate a unique, sorted configuration of endpoint pairs.

        Returns:
            A list of (endpoint_token_0, endpoint_token_1) pairs.
            Each endpoint token is in [0, 2 * num_pins - 1].

        Raises:
            ValueError: If the requested configuration exceeds the
                available number of unique valid strings.
        """
        count = self._sample_string_count()

        if count > self._max_unique_strings:
            raise ValueError(
                f"Requested {count} strings, but only "
                f"{self._max_unique_strings} unique valid strings exist."
            )

        # Sampling without replacement prevents duplicate strings,
        # including when the requested count is large.
        pairs = self._rng.sample(self._valid_pairs, count)

        # Each pair is already canonical; sorting the pairs establishes
        # a deterministic sequence representation.
        pairs.sort()

        return pairs

    def set_seed(self, seed: int) -> None:
        self._rng = random.Random(seed)
