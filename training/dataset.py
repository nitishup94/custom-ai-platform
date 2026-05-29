"""Training sample construction from raw text."""

from __future__ import annotations

import torch
from torch import Tensor

from model.tokenizer import ByteTokenizer


def build_lm_sequence(
    tokenizer: ByteTokenizer,
    user_text: str,
    assistant_text: str,
    max_seq_len: int,
) -> Tensor:
    """Create a single concatenated LM sequence with specials.

    Layout: ``<bos> user <sep> assistant <eos>``. If the packed length exceeds
    ``max_seq_len``, **user** and **assistant** byte-token sequences are each
    prefix-truncated so the separator and end markers always remain (no
    whole-sequence left slice that could drop ``<sep>``).
    """
    user_ids = tokenizer.encode(user_text, add_special_tokens=False)
    answer_ids = tokenizer.encode(assistant_text, add_special_tokens=False)
    overhead = 3
    max_payload = max_seq_len - overhead
    if max_payload < 0:
        msg = "max_seq_len too small for bos + user + sep + assistant + eos"
        raise ValueError(msg)

    u_len, a_len = len(user_ids), len(answer_ids)

    if a_len == 0:
        if u_len > max_payload:
            user_ids = user_ids[:max_payload]
    elif u_len > 0 and a_len > 0 and max_payload < 2:
        msg = "max_seq_len too small for non-empty user and assistant with structure tokens"
        raise ValueError(msg)
    elif u_len + a_len > max_payload:
        la_cap = min(a_len, max_payload - 1)
        lu_cap = max_payload - la_cap
        if u_len <= lu_cap:
            lu_use = u_len
            la_use = min(a_len, max_payload - lu_use)
        else:
            lu_use = lu_cap
            la_use = max_payload - lu_use
        user_ids = user_ids[:lu_use]
        answer_ids = answer_ids[:la_use]

    core: list[int] = (
        [tokenizer.bos_id]
        + user_ids
        + [tokenizer.sep_id]
        + answer_ids
        + [tokenizer.eos_id]
    )
    if len(core) > max_seq_len:
        msg = "internal packing error: sequence still exceeds max_seq_len"
        raise ValueError(msg)
    return torch.tensor([core], dtype=torch.long)


def lm_shifted_targets(input_ids: Tensor) -> Tensor:
    """Targets aligned with ``logits[:, :-1]`` for causal LM."""
    return input_ids[:, 1:].contiguous()
