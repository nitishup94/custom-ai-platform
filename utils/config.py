"""Central application configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from utils.validators import validate_language_tag


class AppConfig(BaseSettings):
    """Runtime configuration with safe defaults for local development."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    app_name: str = Field(default="custom-ai-platform", description="Service name for logs.")
    log_level: str = Field(default="INFO", description="Python logging level name.")
    log_console_enabled: bool = Field(
        default=True,
        description="If true, emit JSON logs to stdout.",
        validation_alias=AliasChoices("LOG_CONSOLE_ENABLED", "LOG_CONSOLE"),
    )
    log_file_path: str = Field(
        default="",
        description="Append JSON logs to this path (relative paths use STORAGE_ROOT). Empty disables file logging.",
        validation_alias=AliasChoices("LOG_FILE_PATH", "LOG_FILE"),
    )
    log_startup_summary: bool = Field(
        default=True,
        description="If true, print one human-readable line to stderr showing active log sinks after setup.",
        validation_alias=AliasChoices("LOG_STARTUP_SUMMARY", "LOG_SHOW_LOGGING_MODES"),
    )

    max_train_field_chars: int = Field(
        default=120_000,
        ge=8192,
        le=500_000,
        description="Max UTF-8 characters per train / dataset input or output field (≈ byte tokens for ASCII).",
        validation_alias=AliasChoices("MAX_TRAIN_FIELD_CHARS", "MAX_TRAIN_TEXT_CHARS"),
    )

    log_training_inputs: bool = Field(
        default=False,
        description="Log each training sample: instruction token count, response token count, instruction preview.",
        validation_alias=AliasChoices("LOG_TRAINING_SAMPLES", "LOG_TRAINING_INPUTS"),
    )

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    uvicorn_reload: bool = Field(default=False)

    storage_root: Path = Field(
        default=Path("."),
        description="Project-relative base for storage paths.",
    )
    chroma_path: Path = Field(default=Path("storage/chroma_data"))
    checkpoint_dir: Path = Field(default=Path("storage/checkpoints"))
    vocab_path: Path = Field(default=Path("storage/vocab/vocab.json"))

    chroma_collection_name: str = Field(default="knowledge")

    model_name: str = Field(default="custom-transformer-v1")
    device: Literal["cpu", "cuda", "auto"] = Field(default="auto")

    d_model: int = Field(default=256, ge=32)
    n_heads: int = Field(default=8, ge=1)
    n_layers: int = Field(default=4, ge=1)
    d_ff: int = Field(default=1024, ge=64)
    max_seq_len: int = Field(default=512, ge=32)
    vocab_size: int = Field(default=512, ge=64)
    dropout: float = Field(default=0.1, ge=0.0, le=0.5)

    learning_rate: float = Field(default=3e-4, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    grad_clip: float = Field(default=1.0, ge=0.0)
    warmup_steps: int = Field(default=100, ge=0)
    max_train_steps: int = Field(default=1_000_000, ge=1)

    checkpoint_every_steps: int = Field(default=10, ge=1)
    epoch_size_steps: int = Field(default=100, ge=1)

    retrieval_top_k: int = Field(default=4, ge=1, le=32)
    generation_temperature: float = Field(default=0.9, ge=0.1, le=5.0)
    generation_top_k: int = Field(default=50, ge=1)

    cors_enabled: bool = Field(default=False)
    cors_allow_origins: str = Field(default="*")

    default_language: str = Field(
        default="en",
        description=(
            "BCP47-style tag applied on POST /train when the client omits `language`. "
            "Set to empty string to store vectors without a language metadata field."
        ),
    )

    @field_validator("default_language", mode="after")
    @classmethod
    def validate_default_language(cls, value: str) -> str:
        """Normalize or clear the default language tag."""
        stripped = (value or "").strip()
        if not stripped:
            return ""
        validated = validate_language_tag(stripped)
        return validated if validated is not None else ""

    def resolved_storage_root(self) -> Path:
        """Return absolute storage root."""
        return self.storage_root.expanduser().resolve()

    def resolved_chroma_path(self) -> Path:
        """Return absolute Chroma persistence directory."""
        path = self.chroma_path
        if not path.is_absolute():
            path = self.resolved_storage_root() / path
        return path.expanduser().resolve()

    def resolved_checkpoint_dir(self) -> Path:
        """Return absolute checkpoint directory."""
        path = self.checkpoint_dir
        if not path.is_absolute():
            path = self.resolved_storage_root() / path
        return path.expanduser().resolve()

    def resolved_vocab_path(self) -> Path:
        """Return absolute vocab file path."""
        path = self.vocab_path
        if not path.is_absolute():
            path = self.resolved_storage_root() / path
        return path.expanduser().resolve()

    def resolved_log_file_path(self) -> Path | None:
        """Return absolute log file path, or ``None`` when file logging is disabled."""
        raw = (self.log_file_path or "").strip()
        if not raw:
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = self.resolved_storage_root() / path
        return path.expanduser().resolve()

    def validate_model_dims(self) -> None:
        """Ensure transformer dimensions are consistent."""
        if self.d_model % self.n_heads != 0:
            msg = "d_model must be divisible by n_heads"
            raise ValueError(msg)
