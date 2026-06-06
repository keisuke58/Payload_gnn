#!/usr/bin/env bash
# Lightweight sanity checks — no Abaqus, no training, no cluster.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> [1/2] H3 fairing spec validation"
python scripts/validate_h3_specs.py

echo "==> [2/2] Unit tests"
python -m pytest tests/ -q --tb=line

echo "==> Done. Full pipeline: see README Quick Start + docs/ARCHITECTURE.md"