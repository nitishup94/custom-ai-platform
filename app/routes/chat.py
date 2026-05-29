"""OpenAI-compatible chat completions."""

from __future__ import annotations

import time
import uuid

import torch
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.container import AppContainer, get_container
from model.tokenizer import ByteTokenizer
from training.dataset import build_lm_sequence
from utils.token_window import clip_byte_token_ids
from utils.validators import validate_language_tag

router = APIRouter(tags=["chat"])


class ChatMessage(BaseModel):
    """Single chat message."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """Subset of OpenAI chat completion schema."""

    messages: list[ChatMessage] = Field(..., min_length=1)
    max_tokens: int = Field(
        default=100,
        ge=1,
        le=20000,
        description="Upper bound on new tokens; also capped by model MAX_SEQ_LEN minus prompt length.",
    )
    temperature: float | None = Field(default=None)
    language: str | None = Field(
        default=None,
        description="If set, Chroma retrieval is restricted to documents stored with this language tag.",
    )


class ChatChoiceMessage(BaseModel):
    """Assistant message payload."""

    role: str
    content: str


class ChatChoice(BaseModel):
    """Single completion choice."""

    index: int
    message: ChatChoiceMessage
    finish_reason: str


class ChatUsage(BaseModel):
    """Approximate token accounting."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-shaped response."""

    id: str
    object: str
    created: int
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage


def _last_user_message(messages: list[ChatMessage]) -> str:
    """Return the most recent user message content."""
    for message in reversed(messages):
        if message.role.lower() == "user":
            return message.content.strip()
    msg = "messages must include at least one user role entry"
    raise ValueError(msg)


def _build_context(docs: list[dict[str, object]], user_prompt: str) -> str:
    """Render retrieved snippets into a plain-text system prefix."""
    if not docs:
        return f"User:\n{user_prompt}"
    parts: list[str] = ["Retrieved knowledge:"]
    for idx, item in enumerate(docs, start=1):
        doc = str(item.get("document", ""))
        parts.append(f"[{idx}] {doc}")
    parts.append(f"User request:\n{user_prompt}")
    return "\n\n".join(parts)


def _truncate_context_for_generation(
    tokenizer: ByteTokenizer,
    context: str,
    max_seq_len: int,
    max_new_tokens: int,
) -> str:
    """Left-truncate *context* so ``prompt + up to max_new_tokens`` fits ``max_seq_len``.

    Prompt layout after ``build_lm_sequence`` and dropping trailing ``eos`` is
    ``bos + user_tokens + sep`` (length ``len(user_tokens) + 2``).
    """
    gen_reserve = min(max_new_tokens, max(1, max_seq_len - 3))
    max_user_tokens = max_seq_len - 2 - gen_reserve
    if max_user_tokens < 1:
        max_user_tokens = 1
    tokens = tokenizer.encode(context, add_special_tokens=False)
    if len(tokens) <= max_user_tokens:
        return context
    tail = tokens[-max_user_tokens:]
    return tokenizer.decode(tail, skip_special_tokens=True)


@router.post("/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(
    body: ChatCompletionRequest,
    container: AppContainer = Depends(get_container),
) -> ChatCompletionResponse:
    """Generate a continuation using retrieval-augmented prompting."""
    try:
        user_prompt = _last_user_message(body.messages)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not user_prompt:
        raise HTTPException(status_code=422, detail="user message content must not be empty")

    try:
        language = validate_language_tag(body.language)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    ids = torch.tensor(
        [clip_byte_token_ids(container.tokenizer, user_prompt, container.config.max_seq_len)],
        dtype=torch.long,
        device=container.device,
    )
    query_vec = container.trainer.inference.retrieval_embedding(ids)
    hits = container.knowledge.query(
        embedding=query_vec.detach().cpu().numpy().tolist()[0],
        top_k=container.config.retrieval_top_k,
        language=language,
    )

    context = _build_context(hits, user_prompt)
    context = _truncate_context_for_generation(
        container.tokenizer,
        context,
        container.config.max_seq_len,
        body.max_tokens,
    )

    prompt_ids = build_lm_sequence(
        container.tokenizer,
        context,
        "",
        max_seq_len=container.config.max_seq_len,
    ).to(container.device)
    prompt_ids = prompt_ids[:, :-1]

    max_seq = container.config.max_seq_len
    room = max_seq - int(prompt_ids.size(1))
    effective_max_new = min(body.max_tokens, max(1, room))

    temperature = body.temperature if body.temperature is not None else container.config.generation_temperature
    if temperature < 0.1 or temperature > 5.0:
        raise HTTPException(status_code=422, detail="temperature must be between 0.1 and 5.0")
    generated = container.trainer.inference.generate(
        prompt_ids,
        max_new_tokens=effective_max_new,
        temperature=temperature,
        top_k=container.config.generation_top_k,
    )
    new_tokens = generated[0, prompt_ids.size(1) :].tolist()
    text = container.tokenizer.decode(new_tokens, skip_special_tokens=True)

    eos_id = container.tokenizer.eos_id
    if new_tokens and new_tokens[-1] == eos_id:
        finish_reason = "stop"
    elif len(new_tokens) >= effective_max_new:
        finish_reason = "length"
    else:
        finish_reason = "stop"

    created = int(time.time())
    response = ChatCompletionResponse(
        id=str(uuid.uuid4()),
        object="chat.completion",
        created=created,
        model=container.config.model_name,
        choices=[
            ChatChoice(
                index=0,
                message=ChatChoiceMessage(role="assistant", content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=ChatUsage(
            prompt_tokens=int(prompt_ids.numel()),
            completion_tokens=len(new_tokens),
            total_tokens=int(prompt_ids.numel()) + len(new_tokens),
        ),
    )
    return response
