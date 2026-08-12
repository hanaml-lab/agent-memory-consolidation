#!/usr/bin/env bash
# Regenerate every table and figure in the paper from the seeded configuration.
set -euo pipefail
cd "$(dirname "$0")/.."
export TIDE_OUT="${TIDE_OUT:-results}"
python scripts/paper/make_data.py
python scripts/paper/figures.py
echo "tables in $TIDE_OUT/tables, figures in $TIDE_OUT/figures"
