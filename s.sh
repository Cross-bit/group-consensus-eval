#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/projects/thesis"

rsync -azP --delete \
  -e "ssh -i $HOME/.ssh/google_compute_engine" \
  --exclude 'cache/' \
  --exclude 'cache/**' \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude 'venv/' \
  --exclude '.venv/' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache/' \
  --exclude '.mypy_cache/' \
  --exclude '.idea/' \
  --exclude '.vscode/' \
  --exclude 'results/' \
  --exclude 'logs/' \
  --exclude '*.log' \
  --exclude 'restaurant_data/**' \
  --exclude '*.pkl' \
  ./analysis/ \
  cross-bit@34.158.184.63:~/analysis/