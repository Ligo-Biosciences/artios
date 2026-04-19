"""Backward-compat shim — real module moved to ``artios.data.structure.tokenizer``.

Exists so that old pickled test data (which references
``artios.data.tokenizer.TokenArray`` in its class metadata) can still be
unpickled. New code should import directly from
``artios.data.structure.tokenizer``.

TODO: regenerate pickled test fixtures and delete this shim.
"""

from artios.data.structure.tokenizer import *  # noqa: F401, F403
