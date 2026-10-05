#!/usr/bin/env bash
# One-shot setup for the NIDAR drone RL project.
# Works in Git Bash / WSL (Windows), Linux and macOS.
#
# Usage:
#   bash setup.sh                      # clone + venv + install
#   MODEL_URL=<zip url> bash setup.sh  # also download trained models (optional)
set -euo pipefail

REPO_URL="https://github.com/Techyshashi/drone_rl.git"
DIR="drone_rl"
VENV=".venv"

# --- 1. Pre-flight checks ---------------------------------------------------
command -v git >/dev/null 2>&1 || { echo "git not found. Install it from https://git-scm.com"; exit 1; }

PY=""
for c in python3 python py; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c "import sys; sys.exit(sys.version_info < (3, 9))" 2>/dev/null; then
    PY="$c"; break
  fi
done
[ -n "$PY" ] || { echo "Python 3.9+ not found. Install it from https://www.python.org"; exit 1; }
echo "[ok] using $($PY --version)"

# --- 2. Clone or update -----------------------------------------------------
if [ -d "$DIR/.git" ]; then
  echo "[git] $DIR exists, pulling latest"
  git -C "$DIR" pull --ff-only
else
  echo "[git] cloning $REPO_URL"
  git clone "$REPO_URL" "$DIR"
fi
cd "$DIR"

# --- 3. Virtual environment -------------------------------------------------
if [ ! -d "$VENV" ]; then
  echo "[venv] creating $VENV"
  "$PY" -m venv "$VENV"
fi

if [ -f "$VENV/Scripts/activate" ]; then   # Windows (Git Bash)
  # shellcheck disable=SC1091
  source "$VENV/Scripts/activate"
else                                        # Linux / macOS / WSL
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi
python -m pip install --upgrade pip

# --- 4. Dependencies --------------------------------------------------------
REQ="$(find . -name requirements.txt -not -path "./$VENV/*" | head -n 1)"
if [ -n "$REQ" ]; then
  echo "[pip] installing from $REQ"
  pip install -r "$REQ"
else
  echo "[pip] no requirements.txt found, installing defaults"
  pip install numpy mujoco gymnasium "stable-baselines3[extra]" torch
fi

# --- 5. Optional: download trained models -----------------------------------
# Models (*.zip, *.pkl) are not in git. Host them (e.g. a GitHub Release) and pass MODEL_URL.
if [ -n "${MODEL_URL:-}" ]; then
  echo "[models] downloading $MODEL_URL"
  curl -L --fail -o models.zip "$MODEL_URL"
  "$PY" -c "import zipfile; zipfile.ZipFile('models.zip').extractall('models')"
  rm -f models.zip
  echo "[models] extracted to ./models"
fi

# --- 6. Done ----------------------------------------------------------------
RUNNER="$(find . -name run_policy_gui.py -not -path "./$VENV/*" | head -n 1)"
echo
echo "Setup complete."
echo "Activate later with:"
if [ -f "$VENV/Scripts/activate" ]; then echo "  source $DIR/$VENV/Scripts/activate"; else echo "  source $DIR/$VENV/bin/activate"; fi
if [ -n "$RUNNER" ]; then
  echo "Run the policy (from $(dirname "$RUNNER")):"
  echo "  python run_policy_gui.py --run <folder with ppo_nidar.zip + vecnorm.pkl> --goal 3"
fi