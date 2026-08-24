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

## Direct project packages

These are all third-party packages imported by the Python source and training
notebook, plus the Jupyter application used to run the notebook. Python
standard-library modules and packages installed only as transitive dependencies
are not listed.

| Package installed with pip | Imported or used as | Purpose | Installed version |
| --- | --- | --- | --- |
| `numpy` | `numpy` | Array processing and radar data structures | 1.26.4 |
| `scipy` | `scipy.fft` | Range and Doppler FFT processing | 1.11.4 |
| `pyserial` | `serial` | Radar command-UART access and port discovery | 3.5 |
| `pyqtgraph` | `pyqtgraph` | Live plots and 3D visualization | 0.14.0 |
| `PySide6` | `PySide6` and `pyqtgraph.Qt` | Qt GUI backend | 6.11.1 |
| `PyOpenGL` | `OpenGL` and `pyqtgraph.opengl` | OpenGL support for 3D views | 3.1.7 |
| `torch` | `torch` | Live classifier inference and model training | 2.13.0 |
| `scikit-learn` | `sklearn` | Training metrics and dataset splitting | 1.9.0 |
| `matplotlib` | `matplotlib` | Training plots | 3.6.3 |
| `onnx` | `onnx` | Model export and parity validation | 1.22.0 |
| `joblib` | `joblib` | Load and save classifier calibration artifacts | 1.5.3 |
| `openradar` | `mmwave.dsp` | OpenRadar DSP backend validation | 1.0.0 (pin: `65bcd628`) |
| `jupyter` | `jupyter` command | Notebook application bundle | 1.1.1 |
| `jupyterlab` | `jupyter lab` command | UI used to open `classification.ipynb` | 4.6.3 |
| `pytest` | `pytest` | Automated test runner used below | 7.4.4 |

The rebuilt environment uses `--system-site-packages`, as recommended for the
Jetson deployment. Consequently, NumPy, SciPy, pyserial, PyOpenGL, and
Matplotlib currently resolve from the compatible system Python installation,
as does `pytest`; most other packages, including OpenRadar, are stored directly
under `.venv`. Use the `.venv` Python interpreter so both package locations are
available.

The environment audit originally found one missing runtime dependency:
`openradar`. It is now installed from the repository's pinned revision and is
imported as `mmwave.dsp` in `rawdatacapture/openradar_backend.py`. Keep using
the pin rather than an unversioned package so that its DSP API remains
reproducible. `joblib` and `pytest` currently resolve through other/system
packages, but they are listed explicitly because the project uses them
directly.

The notebook's `google.colab` import is only for mounting Google Drive when the
notebook runs in Colab. Skip that first cell when running locally; it is not a
`.venv` dependency.

## Rebuild `.venv`

From the repository root:

```bash
git --version
python3 -m venv --clear --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  "numpy<3" "scipy<2" \
  "pyqtgraph>=0.14,<0.15" "PySide6>=6.8,<7" "PyOpenGL>=3.1,<4" \
  pyserial torch "scikit-learn>=1.4,<2" matplotlib onnx joblib jupyter pytest \
  "openradar @ git+https://github.com/PreSenseRadar/OpenRadar.git@65bcd6287af31685acf8b0c32f4505e0f6faab94"
```

Git is required because the pinned OpenRadar dependency is installed directly
from its upstream repository.

PyTorch is platform-specific. On a Jetson with JetPack, prefer NVIDIA's
supported PyTorch wheel for that JetPack release if GPU inference or training
is required. The ordinary PyPI package currently imports and passes this
project's inference tests on CPU, but CUDA availability depends on the host's
driver and JetPack combination.

GPU inference on Jetson additionally uses NVIDIA's TensorRT Python bindings and
the CUDA runtime bindings. Install the TensorRT packages that match JetPack
(the application recommends the `python3-libnvinfer` and `libnvinfer-bin` apt
packages) and install `cuda-python` if `from cuda.bindings import runtime` does
not import. These packages are optional on non-Jetson hosts, where the
application uses the PyTorch CPU backend.

## Verify the installation

Check every direct dependency:

```bash
python - <<'PY'
import matplotlib
import numpy
import onnx
import joblib
import pyqtgraph
import pyqtgraph.opengl
import scipy
import serial
import sklearn
import torch
import mmwave.dsp
from OpenGL import GL
from PySide6 import QtCore, QtWidgets

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
python run.py --display combined
```

Start the training notebook:

```bash
jupyter lab classification.ipynb
```

For radar wiring, capture modes, calibration, output fields, training, and
replay details, see [`rawdatacapture/User guide.md`](rawdatacapture/User%20guide.md).
