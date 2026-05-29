"""Custom tokenizer with JSON persistence (byte-level + optional merge table)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from utils.logger import get_logger

_LOGGER = get_logger("tokenizer")

_SPECIAL_NAMES: Final[tuple[str, ...]] = ("<pad>", "<unk>", "<bos>", "<eos>", "<sep>")


@dataclass(frozen=True)
class VocabSpec:
    """Loaded vocabulary specification."""

    tokenizer_type: str
    specials: dict[str, int]
    byte_offset: int
    reserved_vocab_size: int
    merges: list[list[int]]


class ByteTokenizer:
    """Byte-level tokenizer with fixed special tokens."""

    def __init__(self, spec: VocabSpec, model_vocab_size: int) -> None:
        """Attach tokenizer to model vocabulary size."""
        self._spec = spec
        self._model_vocab_size = model_vocab_size
        self._stoi: dict[str, int] = dict(spec.specials)
        self._unk = spec.specials["<unk>"]
        self._pad = spec.specials["<pad>"]
        self._bos = spec.specials["<bos>"]
        self._eos = spec.specials["<eos>"]
        self._sep = spec.specials["<sep>"]
        self._byte_offset = spec.byte_offset

    @classmethod
    def from_path(cls, path: Path, model_vocab_size: int) -> ByteTokenizer:
        """Load tokenizer from *path* JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        spec = _parse_vocab(data)
        return cls(spec, model_vocab_size)

    @property
    def pad_id(self) -> int:
        """Padding token id."""
        return self._pad

    @property
    def bos_id(self) -> int:
        """Beginning-of-sequence id."""
        return self._bos

    @property
    def eos_id(self) -> int:
        """End-of-sequence id."""
        return self._eos

    @property
    def sep_id(self) -> int:
        """Separator id between prompt and answer in training."""
        return self._sep

    @property
    def vocab_size_effective(self) -> int:
        """Maximum id + 1 required by the tokenizer tables."""
        return min(self._spec.reserved_vocab_size, self._model_vocab_size)

    def byte_token_high_exclusive(self) -> int:
        """Upper bound (exclusive) for byte-derived token ids."""
        return min(self._byte_offset + 256, self._model_vocab_size)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode *text* to token ids (UTF-8 bytes mapped to ids)."""
        raw = text.encode("utf-8")
        ids: list[int] = []
        if add_special_tokens:
            ids.append(self._bos)
        for byte in raw:
            token_id = self._byte_offset + int(byte)
            if token_id >= self._model_vocab_size:
                ids.append(self._unk)
            else:
                ids.append(token_id)
        if add_special_tokens:
            ids.append(self._eos)
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token ids to UTF-8 string."""
        bytes_out: bytearray = bytearray()
        special_values = set(self._spec.specials.values())
        for token_id in ids:
            if skip_special_tokens and token_id in special_values:
                continue
            if self._byte_offset <= token_id < self._byte_offset + 256:
                bytes_out.append(int(token_id - self._byte_offset))
            elif token_id == self._unk:
                continue
        return bytes_out.decode("utf-8", errors="replace")

    def save(self, path: Path) -> None:
        """Persist vocabulary JSON to *path*."""
        payload: dict[str, Any] = {
            "schema_version": 1,
            "tokenizer_type": self._spec.tokenizer_type,
            "specials": self._spec.specials,
            "byte_offset": self._spec.byte_offset,
            "reserved_vocab_size": self._spec.reserved_vocab_size,
            "merges": self._spec.merges,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        _LOGGER.info("Saved vocabulary", extra={"extra_fields": {"path": str(path)}})


def _parse_vocab(data: dict[str, Any]) -> VocabSpec:
    """Parse raw JSON dict into VocabSpec."""
    tokenizer_type = str(data.get("tokenizer_type", "byte"))
    specials_raw = data.get("specials", {})
    specials: dict[str, int] = {str(k): int(v) for k, v in specials_raw.items()}
    for name in _SPECIAL_NAMES:
        if name not in specials:
            msg = f"missing special token {name} in vocab.json"
            raise ValueError(msg)
    byte_offset = int(data.get("byte_offset", 5))
    reserved = int(data.get("reserved_vocab_size", 512))
    merges_raw = data.get("merges", [])
    merges: list[list[int]] = []
    for item in merges_raw:
        if isinstance(item, list) and len(item) == 2:
            merges.append([int(item[0]), int(item[1])])
    return VocabSpec(
        tokenizer_type=tokenizer_type,
        specials=specials,
        byte_offset=byte_offset,
        reserved_vocab_size=reserved,
        merges=merges,
    )


def ensure_vocab_file(path: Path, model_vocab_size: int) -> None:
    """Create a starter vocab file if missing."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = VocabSpec(
        tokenizer_type="byte",
        specials={
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
            "<sep>": 4,
        },
        byte_offset=5,
        reserved_vocab_size=max(512, model_vocab_size),
        merges=[],
    )
    ByteTokenizer(spec, model_vocab_size).save(path)
    _LOGGER.info("Created default vocabulary", extra={"extra_fields": {"path": str(path)}})
