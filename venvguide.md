# Radar-UREx Environment Guide

## Activate the environment

The project virtual environment is `.venv` in the repository root. Activate it
before running the capture application, tests, or training notebook:

```bash
source .venv/bin/activate
```

Confirm that the environment is active with:

```bash
python --version
python -m pip --version
```

Both commands should resolve to paths under this repository's `.venv`.

The environment audited on 24 August 2026 uses Python 3.12.3 and pip 26.2.1.
The version table below records the versions that were actually imported by
that environment, not merely the versions requested from pip.

## Direct project packages

These are all third-party packages imported by the Python source and training
notebook, plus the Jupyter application used to run the notebook. Python
standard-library modules and packages installed only as transitive dependencies
are not listed.

| Distribution or system package | Imported or used as | Purpose | Installed version |
| --- | --- | --- | --- |
| `numpy` | `numpy` | Array processing and radar data structures | 1.26.4 |
| `scipy` | `scipy.fft` | Range and Doppler FFT processing | 1.11.4 |
| `pyserial` | `serial` | Radar command-UART access and port discovery | 3.5 |
| `pyqtgraph` | `pyqtgraph` | Live plots and 3D visualization | 0.14.0 |
| `PySide6` | `PySide6` and `pyqtgraph.Qt` | Qt GUI backend | 6.11.1 |
| `PyOpenGL` | `OpenGL` and `pyqtgraph.opengl` | OpenGL support for 3D views | 3.1.7 |
| `torch` | `torch` | Live classifier inference and model training | 2.13.0+cu130 (distribution 2.13.0) |
| `scikit-learn` | `sklearn` | Calibration loading, DBSCAN, training metrics, and dataset splitting | 1.6.1 |
| `matplotlib` | `matplotlib` | Training plots | 3.6.3 |
| `onnx` | `onnx` | Model export and parity validation | 1.22.0 |
| `joblib` | `joblib` | Load and save classifier calibration artifacts | 1.5.3 |
| `jupyter` | `jupyter` command | Notebook application bundle | 1.1.1 |
| `jupyterlab` | `jupyter lab` command | UI used to open `classification.ipynb` | 4.6.3 |
| `pytest` | `pytest` | Automated test runner used below | 7.4.4 |
| `cuda-bindings` | `cuda.bindings.runtime` | CUDA APIs for the Jetson TensorRT backend | 13.3.1 |
| `TensorRT` (JetPack package) | `tensorrt` | FP16 classifier build and inference on Jetson | 10.16.2.10 |

The GUI also needs the native XCB cursor and OpenGL libraries. The audited host
has `libxcb-cursor0` 0.1.4-1build1 and `libgl1` 1.7.0-1build1. On Ubuntu or
Jetson Linux, install the native prerequisites with:

```bash
sudo apt install libxcb-cursor0 libgl1
```

The rebuilt environment uses `--system-site-packages`, as recommended for the
Jetson deployment. Consequently, NumPy, SciPy, pyserial, PyOpenGL, and
Matplotlib currently resolve from the compatible system Python installation,
as does `pytest`; most other packages are stored directly under `.venv`.
TensorRT 10.16.2.10 resolves from the JetPack system packages, while CUDA
bindings 13.3.1 resolve from `.venv`. Use the `.venv` Python interpreter so
both package locations are available.

`scikit-learn` must remain at 1.6.1 while using the bundled
`model_weights/calibration.joblib`: that is the version which serialized the
calibrator. Loading it with a different version produces an
`InconsistentVersionWarning` and is not a supported deployment configuration.
`joblib` and `pytest` are listed explicitly because the project uses them
directly even when another package or the system installation also provides
them.

The notebook's `google.colab` import is only for mounting Google Drive when the
notebook runs in Colab. Skip that first cell when running locally; it is not a
`.venv` dependency.

## Rebuild `.venv`

From the repository root:

```bash
python3 -m venv --clear --system-site-packages .venv
source .venv/bin/activate
python -m pip install "pip==26.2.1"
python -m pip install \
  "numpy==1.26.4" "scipy==1.11.4" "pyserial==3.5" \
  "pyqtgraph==0.14.0" "PySide6==6.11.1" "PyOpenGL==3.1.7" \
  "torch==2.13.0" "scikit-learn==1.6.1" "matplotlib==3.6.3" \
  "onnx==1.22.0" "joblib==1.5.3" "jupyter==1.1.1" \
  "jupyterlab==4.6.3" "pytest==7.4.4"
```

PyTorch is platform-specific. On a Jetson with JetPack, prefer NVIDIA's
supported PyTorch wheel for that JetPack release if GPU inference or training
is required. The ordinary PyPI package currently imports and passes this
project's inference tests on CPU, but CUDA availability depends on the host's
driver and JetPack combination.

GPU inference on Jetson additionally uses NVIDIA's TensorRT Python bindings and
the CUDA runtime bindings. Install the TensorRT packages that match JetPack
(the audited host uses `python3-libnvinfer` and `libnvinfer-bin`, both at
10.16.2.10), then install the CUDA binding used by this environment:

```bash
python -m pip install "cuda-bindings==13.3.1"
```

The `cuda-python` metapackage is also valid when it supplies the
`cuda.bindings.runtime` module. TensorRT and CUDA bindings are required when
`--classification-device cuda` is selected and optional on non-Jetson hosts,
where the application can use the PyTorch CPU backend.

## Verify the installation

Check every direct dependency:

```bash
python - <<'PY'
from importlib import metadata
from pathlib import Path
import warnings

import joblib
import matplotlib
import numpy
import onnx
import pyqtgraph
import pyqtgraph.opengl
import scipy
import serial
import sklearn
import torch
from OpenGL import GL
from PySide6 import QtCore, QtWidgets
from sklearn.exceptions import InconsistentVersionWarning

expected = {
    "numpy": "1.26.4",
    "scipy": "1.11.4",
    "pyserial": "3.5",
    "pyqtgraph": "0.14.0",
    "PySide6": "6.11.1",
    "PyOpenGL": "3.1.7",
    "torch": "2.13.0",
    "scikit-learn": "1.6.1",
    "matplotlib": "3.6.3",
    "onnx": "1.22.0",
    "joblib": "1.5.3",
    "jupyter": "1.1.1",
    "jupyterlab": "4.6.3",
    "pytest": "7.4.4",
}
for package, wanted in expected.items():
    found = metadata.version(package)
    if found != wanted:
        raise RuntimeError(f"{package}: expected {wanted}, found {found}")
    print(f"{package}=={found}")

warnings.simplefilter("error", InconsistentVersionWarning)
joblib.load("model_weights/calibration.joblib")

if Path("/proc/device-tree/model").is_file():
    from cuda.bindings import runtime as cudart
    import tensorrt
    if metadata.version("cuda-bindings") != "13.3.1":
        raise RuntimeError("Expected cuda-bindings 13.3.1")
    if tensorrt.__version__ != "10.16.2.10":
        raise RuntimeError("Expected TensorRT 10.16.2.10")
    print(f"cuda-bindings=={metadata.version('cuda-bindings')}")
    print(f"TensorRT=={tensorrt.__version__}")

print("All direct project dependencies imported successfully.")
print("CUDA available:", torch.cuda.is_available())
PY
```

Run the automated tests in headless mode:

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

## Run the software

Start live capture from the repository root:

```bash
python -m main.run --display point-cloud-micro-doppler
```

Start the training notebook:

```bash
jupyter lab classification.ipynb
```

For radar wiring, capture modes, calibration, output fields, training, and
replay details, see [`Userguide.md`](Userguide.md).
