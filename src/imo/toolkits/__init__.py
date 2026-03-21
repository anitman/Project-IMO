"""Training toolkits — pluggable adapters for different training backends."""

from imo.toolkits.base import ToolkitCapability, TrainingToolkit
from imo.toolkits.registry import ToolkitRegistry, get_default_registry

__all__ = [
    "TrainingToolkit",
    "ToolkitCapability",
    "ToolkitRegistry",
    "get_default_registry",
]
