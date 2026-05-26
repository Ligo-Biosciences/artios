"""Shared fixtures for the test suite.

The golden-sample pattern: tests that need a parsed `AnnotatedStructure`
load it from `tests/fixtures/structures/<name>.npz` via the
`load_structure` fixture rather than re-parsing a CIF each run. This
makes tests fast and forces the `.save()`/`.load()` contract to be
exercised continuously.

Until a given golden fixture exists on disk, tests requesting it skip
cleanly rather than error. See `plans/testing_strategy.md` for the
intended fixture set.

Torch compilation is disabled for the entire suite via the environment
variable set below. Tests must run without `torch.compile` / Dynamo —
we test model behaviour, not compiler correctness, and dynamo's cross-test
guard caching has been a repeat source of flakes. See `tests/CLAUDE.md`.
"""

import os

# Must be set before torch or artios modules are imported by any test module.
# conftest is the first thing pytest loads, so setting it here propagates.
os.environ["TYPECHECK"] = "true"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from pathlib import Path  # noqa: E402
from typing import Callable  # noqa: E402

import pytest  # noqa: E402

from artios.data.structure import AnnotatedStructure  # noqa: E402

FIXTURES_DIR = Path(__file__).parent / "fixtures"
STRUCTURES_DIR = FIXTURES_DIR / "structures"


@pytest.fixture
def load_structure() -> Callable[[str], AnnotatedStructure]:
    """Load a golden AnnotatedStructure fixture by name.

    Skips the calling test if the fixture doesn't exist on disk yet.

    Usage:
        def test_something(load_structure):
            s = load_structure("protein_only")
            ...
    """

    def _load(name: str) -> AnnotatedStructure:
        path = STRUCTURES_DIR / f"{name}.npz"
        if not path.exists():
            pytest.skip(f"golden fixture {path} not available yet")
        return AnnotatedStructure.load(path)

    return _load


@pytest.fixture
def fixtures_dir() -> Path:
    """Absolute path to `tests/fixtures/`."""
    return FIXTURES_DIR
