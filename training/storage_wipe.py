"""Remove weight checkpoints and temp files from the checkpoint directory."""

from __future__ import annotations

from pathlib import Path


def remove_model_checkpoints(checkpoint_dir: Path) -> list[str]:
    """Delete ``latest.pt``, ``epoch_*.pt``, and ``*.tmp`` under *checkpoint_dir*."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    removed: list[str] = []
    for name in ("latest.pt",):
        path = checkpoint_dir / name
        if path.is_file():
            path.unlink()
            removed.append(name)
    for path in sorted(checkpoint_dir.glob("epoch_*.pt")):
        if path.is_file():
            path.unlink()
            removed.append(path.name)
    for path in sorted(checkpoint_dir.glob("*.tmp")):
        if path.is_file():
            path.unlink()
            removed.append(path.name)
    return removed
