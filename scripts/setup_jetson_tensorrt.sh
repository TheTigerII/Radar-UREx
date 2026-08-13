#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_python="${repository_root}/.venv/bin/python"
venv_pip="${repository_root}/.venv/bin/pip"

if [[ ! -x "${venv_python}" ]]; then
    echo "Project virtual environment not found: ${venv_python}" >&2
    exit 1
fi

sudo apt-get update
sudo apt-get install -y python3-libnvinfer libnvinfer-bin
"${venv_pip}" install 'cuda-python>=12.6,<13'

site_packages="$("${venv_python}" -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n' "/usr/lib/python3/dist-packages" \
    > "${site_packages}/jetson_system_dist_packages.pth"

"${venv_python}" -c \
    'import tensorrt as trt; from cuda.bindings import runtime; print("TensorRT", trt.__version__, "CUDA bindings ready")'
