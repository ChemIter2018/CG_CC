#!/usr/bin/env bash
# Run the ml_lab test suite.
#
# torch (FT-Transformer) and the GBDT libs (xgboost/lightgbm) ship conflicting
# OpenMP runtimes that segfault when both do real work in one process. So the
# torch test runs in a SEPARATE process from the rest. (The production runner
# ml_lab.run already isolates every model in its own subprocess.)
set -e

PY="${PYTHON:-python}"
cd "$(dirname "$0")/.."

echo "=== group 1: core + GBDT (no torch) ==="
"$PY" -m pytest tests/ --ignore=tests/test_ft_transformer.py "$@"

echo "=== group 2: FT-Transformer (torch only) ==="
"$PY" -m pytest tests/test_ft_transformer.py "$@"

echo "=== all test groups passed ==="
