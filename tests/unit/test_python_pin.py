"""DEPLOY-V24-02: enforce Python pin for the deployment runtime (3.13).

Phase 36-02 dropped the original HOUSE-06 pin `>=3.14.0,!=3.14.1` down to
`>=3.13.0,!=3.13.1` because the `python:3.13-slim` container runtime
is the deployment target and python:3.14-slim is not yet broadly available
(see .planning/phases/36-deployment-python-pin/36-CONTEXT.md decision
"Python 3.14 → 3.13 Downgrade Risk").

The `!=3.13.1` exclusion mirrors the prior `!=3.14.1` rationale: a single
patch release with a known dataclasses bug that NetworkX 3.6.1 trips on.
This test enforces the new pin declaratively against pyproject.toml so
any future relaxation back to `>=3.12` (or letting 3.13.1 in) trips CI.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


@pytest.fixture()
def pin_spec() -> SpecifierSet:
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return SpecifierSet(pyproject["project"]["requires-python"])


def test_python_pin_excludes_3_13_1(pin_spec):
    """DEPLOY-V24-02: 3.13.0 admits, 3.13.1 excludes (dataclasses bug), 3.13.2+ admits."""
    assert "3.13.0" in pin_spec, "3.13.0 must satisfy pin"
    assert "3.13.1" not in pin_spec, (
        "3.13.1 must be excluded — NetworkX 3.6.1 trips a CPython 3.13.1 dataclasses bug"
    )
    assert "3.13.2" in pin_spec, "3.13.2 must satisfy pin"
    assert "3.13.4" in pin_spec, "3.13.4 must satisfy pin"


def test_python_pin_excludes_below_3_13(pin_spec):
    """DEPLOY-V24-02 floor: pin is >=3.13.0 — 3.12 and earlier must NOT satisfy it."""
    assert "3.12.0" not in pin_spec, "3.12 below the >=3.13.0 floor"
    assert "3.11.0" not in pin_spec, "3.11 below the >=3.13.0 floor"


def test_python_pin_matches_runtime(pin_spec):
    """Sanity: the active Python interpreter satisfies the pin (if not, CI broken)."""
    runtime = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    assert runtime in pin_spec, f"Runtime {runtime} does not satisfy pin"
