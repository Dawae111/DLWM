#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
deepspeed src/train/train_dlwm.py "${1:-configs/train/dlwm_stage1_7b.yaml}"
