"""Shared setup for offline tests; not imported by experiment runners."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType

from approval_gate.judge_config import add_judge_arguments

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative_path: str) -> ModuleType:
    """Load a fresh script module without caching it in sys.modules.

    Repeated calls stay independent so patches from one test cannot leak into
    another. Paths are relative to the repository, regardless of working directory.
    """
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load test script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def judge_arguments(*extra: str, model: str = "test-model") -> argparse.Namespace:
    """Parse the production judge options with a test-specific default model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=model)
    parser.add_argument("--reviewer-model", default=None)
    add_judge_arguments(parser)
    return parser.parse_args(extra)
