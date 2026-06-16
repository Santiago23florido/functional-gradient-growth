#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
../.venv/bin/python run.py --config configs/compare_functional.yaml "$@"
