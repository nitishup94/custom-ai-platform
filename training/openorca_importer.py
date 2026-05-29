"""Lazy batched reads from OpenOrca-style Parquet corpora using PyArrow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow.parquet as pq

from utils.logger import get_logger

_LOGGER = get_logger("openorca_importer")


@dataclass(frozen=True)
class ColumnMapping:
    """Resolved text column names for question and answer."""

    question: str
    response: str


def _resolve_columns(schema_names: list[str]) -> ColumnMapping:
    """Pick question/answer columns from common OpenOrca / GPT4-augmented layouts."""
    lower = {n.lower(): n for n in schema_names}
    if "question" in lower and "response" in lower:
        return ColumnMapping(question=lower["question"], response=lower["response"])
    if "instruction" in lower and "response" in lower:
        return ColumnMapping(question=lower["instruction"], response=lower["response"])
    if "input" in lower and "output" in lower:
        return ColumnMapping(question=lower["input"], response=lower["output"])
    if "prompt" in lower and "response" in lower:
        return ColumnMapping(question=lower["prompt"], response=lower["response"])
    msg = f"cannot infer question/response columns from parquet schema: {schema_names}"
    raise ValueError(msg)


class OpenOrcaImporter:
    """Stream batches from a parquet file using ``ParquetFile.iter_batches``."""

    def __init__(self, parquet_path: Path) -> None:
        """Open *parquet_path* for streaming reads."""
        if not parquet_path.is_file():
            msg = f"parquet dataset not found: {parquet_path}"
            raise FileNotFoundError(msg)
        self._path = parquet_path
        self._pf = pq.ParquetFile(str(parquet_path))
        self._mapping = _resolve_columns(self._pf.schema_arrow.names)
        _LOGGER.info(
            "Opened parquet dataset",
            extra={
                "extra_fields": {
                    "path": str(parquet_path),
                    "rows": self.total_rows(),
                    "question_col": self._mapping.question,
                    "response_col": self._mapping.response,
                }
            },
        )

    @property
    def question_column(self) -> str:
        """Resolved question / instruction column name."""
        return self._mapping.question

    @property
    def response_column(self) -> str:
        """Resolved answer / response column name."""
        return self._mapping.response

    def total_rows(self) -> int:
        """Return total row count from file metadata."""
        return int(self._pf.metadata.num_rows)

    def iter_batches(self, batch_size: int, *, start_row: int = 0) -> Iterator[pd.DataFrame]:
        """Yield pandas frames using ``iter_batches`` without loading the whole file.

        *start_row* allows resuming by skipping leading rows (re-reads skipped batches
        from disk while advancing the cursor).
        """
        if batch_size < 1:
            msg = "batch_size must be positive"
            raise ValueError(msg)
        cols = [self._mapping.question, self._mapping.response]
        cursor = 0
        for record_batch in self._pf.iter_batches(batch_size=batch_size, columns=cols):
            pdf = record_batch.to_pandas()
            if pdf.empty:
                continue
            batch_start = cursor
            cursor += len(pdf)
            if cursor <= start_row:
                continue
            if batch_start < start_row:
                pdf = pdf.iloc[start_row - batch_start :].copy()
            if pdf.empty:
                continue
            yield pdf.reset_index(drop=True)
