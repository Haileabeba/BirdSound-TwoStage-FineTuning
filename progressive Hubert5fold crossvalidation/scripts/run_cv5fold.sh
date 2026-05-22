#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
python -m progressive_hubert.cli --experiment cv5fold --config "${1:-configs/cv5fold.yaml}"
