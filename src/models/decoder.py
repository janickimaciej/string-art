from abc import ABC, abstractmethod

import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from configs.model_config import ModelConfig


class IDecoder(nn.Module, ABC):
    """Autoregressive endpoint-token decoder with image cross-attention."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Teacher-forced forward pass.

        Args:
            input_ids: [B, T] decoder input token IDs.
            encoder_hidden_states: [B, N, D] image encoder output.
            attention_mask: [B, T], 1 for valid decoder tokens.
            encoder_attention_mask: [B, N], 1 for valid image tokens.

        Returns:
            logits: [B, T, vocab_size].
        """
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        encoder_hidden_states: torch.Tensor,
        max_strings: int | None = None,
        temperature: float = 1.0,
        greedy: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressively generate endpoint tokens.

        Start with BOS and stop at EOS or the configured length limit.

        Returns:
            sequences: [B, T], including BOS and EOS when generated.
        """
        raise NotImplementedError


class GPT2Decoder(IDecoder):
    """
    GPT-2 decoder with causal self-attention and image cross-attention.

    Input token embeddings are combined with learned string-position
    and endpoint-position embeddings.

    Output logits have shape [B, T, vocab_size].
    """

    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__(config)

        if config.hidden_dim % config.num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads."
            )
        if config.decoder_layers < 1:
            raise ValueError("decoder_layers must be at least 1.")
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.max_positions = config.max_positions
        self.max_supported_strings = min(
            config.max_strings,
            (self.max_positions - 2) // 2,
        )

        # GPT-2 architecture with encoder-decoder cross-attention.
        gpt_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=self.max_positions,
            n_ctx=self.max_positions,
            n_embd=config.hidden_dim,
            n_layer=config.decoder_layers,
            n_head=config.num_heads,
            activation_function="gelu_new",
            resid_pdrop=config.dropout,
            embd_pdrop=config.dropout,
            attn_pdrop=config.dropout,
            add_cross_attention=True,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            use_cache=False,
        )

        self.gpt2 = GPT2LMHeadModel(gpt_config)

        # We provide our own structural position embeddings, so disable
        # GPT-2's learned absolute positional embeddings.
        self.gpt2.transformer.wpe = nn.Embedding(
            max_positions,
            config.hidden_dim,
        )
        nn.init.zeros_(self.gpt2.transformer.wpe.weight)
        self.gpt2.transformer.wpe.weight.requires_grad_(False)

        # BOS uses string position 0. The first endpoint pair uses
        # string position 0, the second pair position 1, and so on.
        self.string_position_embeddings = nn.Embedding(
            self.max_supported_strings + 1,
            config.hidden_dim,
        )

        # Endpoint side within a string: first=0, second=1.
        self.endpoint_position_embeddings = nn.Embedding(
            2,
            config.hidden_dim,
        )

        nn.init.normal_(
            self.string_position_embeddings.weight,
            mean=0.0,
            std=0.02,
        )
        nn.init.normal_(
            self.endpoint_position_embeddings.weight,
            mean=0.0,
            std=0.02,
        )

    def _validate_input_ids(self, input_ids: torch.Tensor) -> None:
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}"
            )
        if input_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError("input_ids must contain integer token IDs.")
        if input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
            raise ValueError("Input batch and sequence must be nonempty.")
        if input_ids.shape[1] > self.max_positions:
            raise ValueError(
                f"Sequence length {input_ids.shape[1]} exceeds "
                f"max_positions={self.max_positions}."
            )
        if input_ids.min().item() < 0:
            raise ValueError("Token IDs cannot be negative.")
        if input_ids.max().item() >= self.config.vocab_size:
            raise ValueError(
                f"Token IDs must be below vocab_size={self.config.vocab_size}."
            )

    def _validate_encoder_states(
        self,
        encoder_hidden_states: torch.Tensor,
    ) -> None:
        if encoder_hidden_states.ndim != 3:
            raise ValueError(
                "encoder_hidden_states must have shape [B, N, D]."
            )
        if encoder_hidden_states.shape[1] < 1:
            raise ValueError("Encoder memory must contain at least one token.")
        if encoder_hidden_states.shape[2] != self.config.hidden_dim:
            raise ValueError(
                f"Expected encoder hidden dimension {self.config.hidden_dim}, "
                f"got {encoder_hidden_states.shape[2]}."
            )
        if not encoder_hidden_states.is_floating_point():
            raise TypeError("Encoder states must be floating-point tensors.")

    def _build_inputs_embeds(
    self,
    input_ids: torch.Tensor,
    position_offset: int = 0,
    ) -> torch.Tensor:
        """Build token + string-position + endpoint-position embeddings."""

        sequence_length = input_ids.shape[1]
        device = input_ids.device

        # Absolute positions in the complete sequence, even when only
        # the newest token is processed using the KV cache.
        positions = torch.arange(
            position_offset,
            position_offset + sequence_length,
            device=device,
        )

        structural_positions = (positions - 1).clamp_min(0)

        string_positions = structural_positions // 2
        endpoint_positions = structural_positions % 2

        string_positions = string_positions.clamp(
            max=self.max_supported_strings
        )

        # BOS occupies sequence position zero.
        is_bos = positions == 0
        string_positions = torch.where(
            is_bos,
            torch.zeros_like(string_positions),
            string_positions,
        )
        endpoint_positions = torch.where(
            is_bos,
            torch.zeros_like(endpoint_positions),
            endpoint_positions,
        )

        token_embeds = self.gpt2.transformer.wte(input_ids)
        string_embeds = self.string_position_embeddings(string_positions)
        endpoint_embeds = self.endpoint_position_embeddings(endpoint_positions)

        return (
            token_embeds
            + string_embeds.unsqueeze(0)
            + endpoint_embeds.unsqueeze(0)
        )

    def _forward_with_cache(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        past_key_values=None,
        position_offset: int = 0,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ):
        inputs_embeds = self._build_inputs_embeds(
            input_ids,
            position_offset=position_offset,
        )

        return self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Teacher-forced forward pass.

        Args:
            input_ids: [B, T] decoder input token IDs.
            encoder_hidden_states: [B, N, D] image encoder output.
            attention_mask: [B, T], 1 for valid decoder tokens.
            encoder_attention_mask: [B, N], 1 for valid image tokens.

        Returns:
            logits: [B, T, vocab_size].
        """
        self._validate_input_ids(input_ids)
        self._validate_encoder_states(encoder_hidden_states)

        batch_size = input_ids.shape[0]

        if encoder_hidden_states.shape[0] != batch_size:
            raise ValueError(
                "Decoder input and encoder memory batch sizes must match."
            )
        if input_ids.device != encoder_hidden_states.device:
            raise ValueError(
                "Decoder input and encoder memory must be on the same device."
            )

        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have shape [B, T].")
            if attention_mask.device != input_ids.device:
                raise ValueError("attention_mask must be on the input device.")

        if encoder_attention_mask is not None:
            if encoder_attention_mask.shape != encoder_hidden_states.shape[:2]:
                raise ValueError(
                    "encoder_attention_mask must have shape [B, N]."
                )
            if encoder_attention_mask.device != input_ids.device:
                raise ValueError(
                    "encoder_attention_mask must be on the input device."
                )

        inputs_embeds = self._build_inputs_embeds(input_ids)

        outputs = self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,
            return_dict=True,
        )

        return outputs.logits

    @torch.no_grad()
    def generate(
        self,
        encoder_hidden_states: torch.Tensor,
        max_strings: int | None = None,
        temperature: float = 1.0,
        greedy: bool = True,
    ) -> torch.Tensor:
        """Generate endpoint tokens using cached autoregressive decoding."""

        self._validate_encoder_states(encoder_hidden_states)

        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        if max_strings is None:
            max_strings = self.max_supported_strings

        if not 1 <= max_strings <= self.max_supported_strings:
            raise ValueError(
                f"max_strings must be between 1 and "
                f"{self.max_supported_strings}."
            )

        device = encoder_hidden_states.device
        batch_size = encoder_hidden_states.shape[0]

        sequences = torch.full(
            (batch_size, 1),
            self.config.bos_token_id,
            dtype=torch.long,
            device=device,
        )

        finished = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=device,
        )

        # The decoder attention mask includes the cached prefix.
        attention_mask = torch.ones_like(sequences)

        past_key_values = None
        max_total_length = 2 * max_strings + 2

        was_training = self.training
        self.eval()

        try:
            for _ in range(max_total_length - 1):
                if past_key_values is None:
                    current_ids = sequences
                    position_offset = 0
                else:
                    current_ids = sequences[:, -1:]
                    position_offset = sequences.shape[1] - 1

                outputs = self._forward_with_cache(
                    input_ids=current_ids,
                    encoder_hidden_states=encoder_hidden_states,
                    past_key_values=past_key_values,
                    position_offset=position_offset,
                    attention_mask=attention_mask,
                )

                past_key_values = outputs.past_key_values
                next_logits = outputs.logits[:, -1, :].float()

                # BOS is never a valid continuation token.
                next_logits[:, self.config.bos_token_id] = float("-inf")

                endpoint_count = sequences.shape[1] - 1

                if endpoint_count >= 2 * max_strings:
                    next_tokens = torch.full(
                        (batch_size,),
                        self.config.eos_token_id,
                        dtype=torch.long,
                        device=device,
                    )
                else:
                    next_logits = next_logits / temperature

                    if greedy:
                        next_tokens = next_logits.argmax(dim=-1)
                    else:
                        probabilities = torch.softmax(next_logits, dim=-1)
                        next_tokens = torch.multinomial(
                            probabilities,
                            num_samples=1,
                        ).squeeze(-1)

                    next_tokens = torch.where(
                        finished,
                        torch.full_like(
                            next_tokens,
                            self.config.eos_token_id,
                        ),
                        next_tokens,
                    )

                sequences = torch.cat(
                    [sequences, next_tokens.unsqueeze(1)],
                    dim=1,
                )

                finished |= next_tokens.eq(self.config.eos_token_id)

                if finished.all():
                    break

                # Prepare a mask for the next iteration. Its length must
                # match the complete cached prefix, including the newest token.
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (batch_size, 1),
                            dtype=attention_mask.dtype,
                            device=device,
                        ),
                    ],
                    dim=1,
                )

        finally:
            self.train(was_training)

        return sequences
