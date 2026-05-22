#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m progressive_hubert.cli --experiment overlap --config configs/local.yaml "$@"
