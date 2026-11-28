#!/bin/bash
# One-time setup for the FAISS baseline: a Python virtualenv with faiss-gpu.
# Run on an Euler LOGIN node (needs internet):  bash scripts/setup_faiss.sh
#
# Uses the faiss-gpu-cu12 wheel, which bundles its own CUDA 12 runtime via pip;
# a newer NVIDIA driver (e.g. from CUDA 13) runs CUDA 12 programs fine.
# Needs Python >= 3.9: if the system python3 is older, `module load` a newer
# Python first (see `module avail python`), or set PYTHON=/path/to/python3.
set -euo pipefail

VENV="${FAISS_VENV:-$HOME/.venvs/vecsearch-faiss}"
PY="${PYTHON:-python3}"

"$PY" - <<'PYCHECK'
import sys
if sys.version_info < (3, 9):
    sys.exit("Python %s is too old for faiss-gpu wheels; module load a Python >= 3.9 "
             "or set PYTHON=..." % sys.version.split()[0])
PYCHECK

if [[ ! -x "$VENV/bin/python" ]]; then
    "$PY" -m venv "$VENV"
fi
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet "faiss-gpu-cu12" numpy
"$VENV/bin/python" -c "import faiss, numpy; print('faiss', faiss.__version__, '| numpy', numpy.__version__)"
echo "FAISS environment ready: $VENV"
echo "(No GPU is visible on the login node; slurm/baselines.sbatch runs it on a GPU node.)"
