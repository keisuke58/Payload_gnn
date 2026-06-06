"""Smoke tests for core module imports (no dataset required)."""

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_import_models_deeponet():
    mod = importlib.import_module("models_deeponet")
    assert hasattr(mod, "DeepONet")


@pytest.mark.gnn
def test_import_models_gnn():
    pytest.importorskip("torch_geometric")
    mod = importlib.import_module("models_gnn")
    assert mod is not None