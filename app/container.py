"""Application-wide dependency container."""

from __future__ import annotations

import torch
from fastapi import Request

from model.tokenizer import ByteTokenizer, ensure_vocab_file
from model.transformer import TransformerLM
from training.trainer import Trainer
from utils.config import AppConfig
from utils.logger import get_logger
from vectorstore.chroma_client import build_chroma_client
from vectorstore.collections import KnowledgeCollection


class AppContainer:
    """Holds long-lived service singletons."""

    def __init__(
        self,
        config: AppConfig,
        tokenizer: ByteTokenizer,
        trainer: Trainer,
        knowledge: KnowledgeCollection,
        device: torch.device,
    ) -> None:
        """Store wired dependencies."""
        self.config = config
        self.tokenizer = tokenizer
        self.trainer = trainer
        self.knowledge = knowledge
        self.device = device
        self.logger = get_logger("container")

    @classmethod
    def bootstrap(cls, config: AppConfig) -> AppContainer:
        """Construct the container graph."""
        config.validate_model_dims()
        log = get_logger("container")
        device = _resolve_device(config)
        log.info("Using device", extra={"extra_fields": {"device": str(device)}})

        config.resolved_checkpoint_dir().mkdir(parents=True, exist_ok=True)
        ensure_vocab_file(config.resolved_vocab_path(), config.vocab_size)
        tokenizer = ByteTokenizer.from_path(config.resolved_vocab_path(), config.vocab_size)

        chroma = build_chroma_client(config)
        knowledge = KnowledgeCollection(chroma, config)

        model = TransformerLM(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )
        trainer = Trainer(model, tokenizer, config, device)
        trainer.load_latest()
        return cls(config=config, tokenizer=tokenizer, trainer=trainer, knowledge=knowledge, device=device)


def _resolve_device(config: AppConfig) -> torch.device:
    """Pick execution device from configuration."""
    if config.device == "cpu":
        return torch.device("cpu")
    if config.device == "cuda":
        if not torch.cuda.is_available():
            msg = "CUDA requested but not available"
            raise RuntimeError(msg)
        return torch.device("cuda")
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device("cpu")


def get_container(request: Request) -> AppContainer:
    """FastAPI dependency that returns the bootstrapped container."""
    container = getattr(request.app.state, "container", None)
    if container is None:
        msg = "application container is not initialized"
        raise RuntimeError(msg)
    return container  # type: ignore[no-any-return]
