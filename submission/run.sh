#!/bin/bash
# Pilots: H -> WW measurement pipeline.
#   run.sh <dataset_dir> <output_dir>  ->  <output_dir>/RESULT.json
# A two-dimensional (m_T, m_ll) template fit for the Higgs mass and the signal strength
# (fit.py, numpy + scipy only). See fit.py for the details.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$2"
python3 "$HERE/fit.py" "$1" "$2"
