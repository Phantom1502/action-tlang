"""Entities package."""
from .schema import (
    AppConfig,
    BaseConfig,
    WindowConfig,
    TrainingConfig,
)

from .loader import (
    load_config,
)

__all__ = [
    "AppConfig",
    "BaseConfig",
    "WindowConfig",
    "TrainingConfig",
    "load_config",
]
