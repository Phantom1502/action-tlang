"""Entities package."""
from .schema import (
    AppConfig,
    BaseConfig,
    WindowConfig,
)

from .loader import (
    load_config,
)

__all__ = [
    "AppConfig",
    "BaseConfig",
    "WindowConfig",
    "load_config",
]
