"""Entities package."""
from .arguments import DataArguments
from .data_module import (
    make_data_module
)
__all__ = [
    "DataArguments",
    "make_data_module",
]
