#!/usr/bin/env bash
# One command setup + launch (Linux/macOS):  bash run.sh
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
[ -d .venv ] || $PY -m venv .venv
source .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q jupyter nbconvert ipykernel
python scripts/configure.py          # optional free keys, Enter to skip
mkdir -p reports
jupyter nbconvert --to notebook --execute CineMind_Launcher.ipynb \
  --output-dir reports --output launcher_executed --ExecutePreprocessor.timeout=-1
echo "Done. Executed notebook: reports/launcher_executed.ipynb"
