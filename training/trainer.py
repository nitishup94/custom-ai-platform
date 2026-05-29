"""Single-step training orchestration."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from model.inference import InferenceEngine
from model.tokenizer import ByteTokenizer
from model.transformer import TransformerLM
from training.checkpoint import latest_checkpoint_path, load_checkpoint, maybe_save_epoch_copy, save_checkpoint
from training.dataset import build_lm_sequence, lm_shifted_targets
from training.optimizer import build_optimizer
from training.scheduler import build_scheduler
from utils.config import AppConfig
from utils.logger import get_logger

_LOGGER = get_logger("trainer")


class Trainer:
    """Coordinates forward/backward passes and checkpointing."""

    def __init__(
        self,
        model: TransformerLM,
        tokenizer: ByteTokenizer,
        config: AppConfig,
        device: torch.device,
    ) -> None:
        """Wire model, tokenizer, and device."""
        self._model = model.to(device)
        self._tokenizer = tokenizer
        self._config = config
        self._device = device
        self._optimizer = build_optimizer(self._model, config)
        self._scheduler = build_scheduler(self._optimizer, config)
        self._global_step = 0
        self._epoch = 0
        self._inference = InferenceEngine(
            self._model,
            pad_id=tokenizer.pad_id,
            eos_id=tokenizer.eos_id,
        )

    @property
    def model(self) -> TransformerLM:
        """Underlying language model."""
        return self._model

    @property
    def inference(self) -> InferenceEngine:
        """Inference helper sharing weights."""
        return self._inference

    @property
    def global_step(self) -> int:
        """Optimizer step counter."""
        return self._global_step

    def load_latest(self) -> None:
        """Load ``latest.pt`` if present."""
        path = latest_checkpoint_path(self._config)
        if path.exists():
            self._global_step, self._epoch = load_checkpoint(path, self._model, self._optimizer)

    def reset_runtime_after_storage_wipe(self) -> None:
        """Zero step counters and rebuild optimizer/scheduler after checkpoints were removed on disk."""
        self._global_step = 0
        self._epoch = 0
        self._optimizer = build_optimizer(self._model, self._config)
        self._scheduler = build_scheduler(self._optimizer, self._config)

    def train_on_sample(self, user_text: str, assistant_text: str) -> float:
        """Run one optimization step and return scalar loss."""
        self._model.train()
        input_ids = build_lm_sequence(
            self._tokenizer,
            user_text,
            assistant_text,
            self._config.max_seq_len,
        ).to(self._device)
        logits = self._model(input_ids)
        targets = lm_shifted_targets(input_ids).to(self._device)
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self._config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._config.grad_clip)
        self._optimizer.step()
        if self._scheduler is not None:
            self._scheduler.step()
        self._global_step += 1

        latest = latest_checkpoint_path(self._config)
        save_checkpoint(
            latest,
            self._model,
            self._optimizer,
            global_step=self._global_step,
            epoch=self._epoch,
        )
        if self._global_step % self._config.checkpoint_every_steps == 0:
            self._epoch += 1
            maybe_save_epoch_copy(
                self._config,
                self._epoch,
                self._model,
                self._optimizer,
                global_step=self._global_step,
            )

        return float(loss.detach().cpu().item())
