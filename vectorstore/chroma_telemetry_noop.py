"""No-op Chroma product telemetry (avoids PostHog ``capture`` API mismatches)."""

from __future__ import annotations

from chromadb.config import System
from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent
from overrides import override


class NoOpProductTelemetry(ProductTelemetryClient):
    """Drop-in replacement for ``chromadb.telemetry.product.posthog.Posthog``."""

    def __init__(self, system: System) -> None:
        """Wire the telemetry client into the Chroma ``System`` graph."""
        super().__init__(system)

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        """Ignore telemetry events."""
        return
