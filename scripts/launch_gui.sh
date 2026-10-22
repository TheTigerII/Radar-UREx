#!/bin/sh

set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
venv_python="$project_dir/.venv/bin/python"

if [ -x "$venv_python" ]; then
    exec "$venv_python" "$project_dir/main/gui.py"
fi

exec python3 "$project_dir/main/gui.py"
