#!/usr/bin/env bash
set -euo pipefail

# Prepare the environment, data, weights and norm stats as documented first.
# Run inside an allocated GPU worker. Extra arguments are forwarded to train.py.
export WANDB_MODE="${WANDB_MODE:-offline}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
python scripts/train.py pi0_forceldm "$@"
