"""Toolkit registry — discovers and manages available training toolkits."""

from __future__ import annotations

from imo.data.dataset_spec import ModelCategory
from imo.toolkits.base import TrainingToolkit


class ToolkitRegistry:
    """Registry of available training toolkits."""

    def __init__(self) -> None:
        self._toolkits: dict[str, TrainingToolkit] = {}

    def register(self, toolkit: TrainingToolkit) -> None:
        """Register a toolkit."""
        self._toolkits[toolkit.info().name] = toolkit

    def get(self, name: str) -> TrainingToolkit | None:
        """Get a toolkit by name."""
        return self._toolkits.get(name)

    def list_all(self) -> list[TrainingToolkit]:
        """List all registered toolkits."""
        return list(self._toolkits.values())

    def list_names(self) -> list[str]:
        """List registered toolkit names."""
        return list(self._toolkits.keys())

    def find_for_category(self, category: ModelCategory) -> list[TrainingToolkit]:
        """Find toolkits that support a given model category."""
        return [t for t in self._toolkits.values() if t.supports_category(category)]

    def find_for_mode(self, mode: str) -> list[TrainingToolkit]:
        """Find toolkits that support a given training mode."""
        return [t for t in self._toolkits.values() if t.supports_mode(mode)]

    def recommend(self, category: ModelCategory, mode: str) -> list[TrainingToolkit]:
        """Recommend toolkits that support both the category and mode."""
        return [
            t
            for t in self._toolkits.values()
            if t.supports_category(category) and t.supports_mode(mode)
        ]


def get_default_registry() -> ToolkitRegistry:
    """Create a registry pre-loaded with all built-in toolkits."""
    from imo.toolkits.builtin.ai_toolkit import AIToolkitAdapter
    from imo.toolkits.builtin.axolotl import AxolotlAdapter
    from imo.toolkits.builtin.diffusers import DiffusersAdapter
    from imo.toolkits.builtin.hf_trainer import HFTrainerAdapter
    from imo.toolkits.builtin.musubi import MusubiAdapter
    from imo.toolkits.builtin.unsloth import UnslothAdapter

    registry = ToolkitRegistry()
    registry.register(HFTrainerAdapter())
    registry.register(UnslothAdapter())
    registry.register(AxolotlAdapter())
    registry.register(DiffusersAdapter())
    registry.register(MusubiAdapter())
    registry.register(AIToolkitAdapter())
    return registry
