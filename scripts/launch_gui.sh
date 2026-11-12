#!/bin/sh

set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
venv_python="$project_dir/.venv/bin/python"
gui_path="$project_dir/aesa_radar/main/gui.py"

if [ -x "$venv_python" ]; then
    exec "$venv_python" -u "$gui_path"
fi

exec python3 -u "$gui_path"
