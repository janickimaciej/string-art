from dataclasses import dataclass


@dataclass(frozen=True)
class TokenSchema:
    """Defines the discrete token space used by the string decoder."""

    num_pins: int = 256
    num_sides: int = 2

    bos_token_id: int = num_pins * num_sides
    eos_token_id: int = bos_token_id + 1

    @property
    def num_endpoint_tokens(self) -> int:
        """Number of tokens representing pin/side endpoint combinations."""
        ...

    @property
    def vocab_size(self) -> int:
        """Total decoder vocabulary size, including special tokens."""
        ...

    def pin_to_token(self, pin: int, side: int) -> int:
        """Convert a pin index and connection side into an endpoint token ID."""
        ...

    def token_to_pin(self, token: int) -> int:
        """Extract the pin index from an endpoint token ID."""
        ...

    def token_to_side(self, token: int) -> int:
        """Extract the connection side from an endpoint token ID."""
        ...

    def is_endpoint_token(self, token: int) -> bool:
        """Return whether the token represents a string endpoint."""
        ...

    def is_special_token(self, token: int) -> bool:
        """Return whether the token is BOS or EOS."""
        ...

    def is_valid_token(self, token: int) -> bool:
        """Return whether the token belongs to the vocabulary."""
        ...
