from typing import Sequence, TypeAlias

from .token_schema import TokenSchema


Endpoint: TypeAlias = tuple[int, int]
String: TypeAlias = tuple[Endpoint, Endpoint]


class StringTokenizer:
    """Converts string-art endpoints/strings to and from decoder token IDs."""

    def __init__(self, schema: TokenSchema) -> None:
        """Initialize the tokenizer with the token-space definition."""
        ...

    def encode_endpoint(
        self,
        endpoint: Endpoint,
    ) -> int:
        """Convert an (pin, side) endpoint into a single token ID."""
        ...

    def decode_endpoint(
        self,
        token: int,
    ) -> Endpoint:
        """Convert an endpoint token ID into a (pin, side) endpoint."""
        ...

    def encode_string(
        self,
        string: String,
    ) -> tuple[int, int]:
        """Convert one string into its two endpoint token IDs."""
        ...

    def decode_string(
        self,
        tokens: Sequence[int],
    ) -> String:
        """Convert two endpoint token IDs into one string."""
        ...

    def encode(
        self,
        strings: Sequence[String],
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        """Convert a sequence of strings into a flat decoder token sequence."""
        ...

    def decode(
        self,
        tokens: Sequence[int],
        stop_at_eos: bool = True,
    ) -> list[String]:
        """Convert a decoder token sequence into a sequence of strings."""
        ...

    def token_to_string_index(
        self,
        token_position: int,
    ) -> int:
        """Return the string index corresponding to a position in the flat sequence."""
        ...

    def token_to_endpoint_index(
        self,
        token_position: int,
    ) -> int:
        """Return whether a sequence position is endpoint 0 or endpoint 1."""
        ...
