#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXPERIMENT_ROOT/.." && pwd)"
exec "$PROJECT_ROOT/.venv/bin/python" -u "$EXPERIMENT_ROOT/main/gui.py"
