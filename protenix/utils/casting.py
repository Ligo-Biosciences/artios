"""Utilities for controlling autocast behavior."""

import functools
from typing import Callable, TypeVar

import torch

F = TypeVar("F", bound=Callable)


def disable_autocast(func: F) -> F:
    """Decorator to disable autocast and run function in fp32.

    Use this on loss computation functions that require higher precision.

    Args:
        func: Function to wrap.

    Returns:
        Wrapped function that runs with autocast disabled.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with torch.autocast(device_type="cuda", enabled=False):
            return func(*args, **kwargs)

    return wrapper  # type: ignore
