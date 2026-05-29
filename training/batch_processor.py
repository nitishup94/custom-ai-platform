"""Map parquet rows to training payloads and invoke Chroma + trainer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from training.category_detector import detect_category
from utils.logger import get_logger
from utils.token_window import clip_byte_token_ids
from utils.training_sample_log import training_sample_extra_fields
from utils.validators import validate_bounded_text, validate_category, validate_language_tag

if TYPE_CHECKING:
    from app.container import AppContainer

_LOGGER = get_logger("batch_processor")


def rows_to_payloads(
    question_col: str,
    response_col: str,
    frame: Any,
    language: str,
) -> list[dict[str, str]]:
    """Build validated training dicts from a dataframe chunk."""
    out: list[dict[str, str]] = []
    for _, row in frame.iterrows():
        q = str(row[question_col]).strip()
        a = str(row[response_col]).strip()
        if not q or not a:
            continue
        category = detect_category(q, a)
        out.append(
            {
                "input": q,
                "output": a,
                "category": category,
                "language": language,
            }
        )
    return out


def train_payload_batch(container: "AppContainer", payloads: list[dict[str, str]]) -> float:
    """Run Chroma indexing + ``train_on_sample`` for each payload; return mean loss."""
    if not payloads:
        return 0.0
    losses: list[float] = []
    for item in payloads:
        question = validate_bounded_text(
            item["input"],
            "input",
            container.config.max_train_field_chars,
        )
        answer = validate_bounded_text(
            item["output"],
            "output",
            container.config.max_train_field_chars,
        )
        category = validate_category(item["category"])
        language = validate_language_tag((item.get("language") or "").strip())
        if language is None and container.config.default_language:
            language = validate_language_tag(container.config.default_language)

        if container.config.log_training_inputs:
            _LOGGER.info(
                "Dataset train row",
                extra={
                    "extra_fields": training_sample_extra_fields(
                        container.tokenizer,
                        question,
                        answer,
                    )
                },
            )

        doc = f"Question: {question}\nAnswer: {answer}"
        cap = container.config.max_seq_len
        q_ids = clip_byte_token_ids(container.tokenizer, question, cap)
        ids = torch.tensor([q_ids], dtype=torch.long, device=container.device)
        embedding = container.trainer.inference.retrieval_embedding(ids)
        vector = embedding.detach().cpu().numpy().tolist()[0]
        container.knowledge.add_document(
            embedding=vector,
            document=doc,
            category=category,
            source="openorca_dataset",
            language=language,
        )
        loss = container.trainer.train_on_sample(question, answer)
        losses.append(loss)
    return sum(losses) / float(len(losses))
