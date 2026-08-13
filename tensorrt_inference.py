from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from inference import (
    DOPPLER_BINS,
    FEATURE_VERSION,
    INPUT_SHAPE_CHW,
    TARGET_GATE_BINS,
    WINDOW_STEPS,
    DroneBirdInference,
    InferenceResult,
    RadarInferenceConfig,
    doppler_cube_to_feature_step,
    normalized_profile_sha256,
)


ENGINE_FORMAT_VERSION = 2
ENGINE_PRECISION = "fp16"
ENGINE_WORKSPACE_BYTES = 256 * 1024 * 1024
ENGINE_PRECISION_POLICY = "tensorrt_fp16_auto_tactics_v2"
PARITY_PROBABILITY_TOLERANCE = 1e-3
DEFAULT_WARMUP_RUNS = 20
DEFAULT_BENCHMARK_RUNS = 200
ONNX_MODEL_NAME = "model.onnx"
PARITY_DATA_NAME = "parity.npz"
JETSON_MODEL_PATH = Path("/proc/device-tree/model")
SYSTEM_DIST_PACKAGES = (
    Path("/usr/lib/python3/dist-packages"),
    Path(f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages"),
)


class TensorRTInferenceError(RuntimeError):
    """Raised when the required Jetson TensorRT backend cannot be used safely."""


@dataclass(frozen=True)
class TensorRTBenchmark:
    warmup_runs: int
    measured_runs: int
    p50_ms: float
    p95_ms: float
    max_ms: float


def is_jetson() -> bool:
    """Return whether the current host is an NVIDIA Jetson platform."""
    if JETSON_MODEL_PATH.is_file():
        try:
            return "nvidia" in JETSON_MODEL_PATH.read_text(
                encoding="utf-8",
                errors="ignore",
            ).lower()
        except OSError:
            pass
    return platform.machine() == "aarch64" and "tegra" in platform.release().lower()


def resolve_classification_device(requested: str) -> str:
    """Resolve auto without silently selecting CPU on a Jetson."""
    normalized = str(requested).strip().lower()
    if normalized not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"Unsupported classification device: {requested}")
    if normalized == "auto":
        return "cuda" if is_jetson() else "cpu"
    return normalized


def create_inference_engine(
    artifact_dir: Path,
    config: RadarInferenceConfig,
    profile_path: Path,
    *,
    device: str = "auto",
    parity_data_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> DroneBirdInference | "TensorRTDroneBirdInference":
    """Create the requested backend; CUDA failures are deliberately fatal."""
    resolved = resolve_classification_device(device)
    if resolved == "cpu":
        return DroneBirdInference(artifact_dir, config, profile_path)
    return TensorRTDroneBirdInference(
        artifact_dir,
        config,
        profile_path,
        parity_data_path=parity_data_path,
        progress_callback=progress_callback,
    )


def _import_tensorrt() -> Any:
    try:
        import tensorrt as trt

        return trt
    except ModuleNotFoundError:
        for path in SYSTEM_DIST_PACKAGES:
            if path.is_dir() and str(path) not in sys.path:
                sys.path.append(str(path))
        try:
            import tensorrt as trt

            return trt
        except ModuleNotFoundError as exc:
            raise TensorRTInferenceError(
                "TensorRT Python bindings are unavailable. Install the matching "
                "Jetson packages with: sudo apt install python3-libnvinfer "
                "libnvinfer-bin"
            ) from exc
        except Exception as exc:
            raise TensorRTInferenceError(
                f"TensorRT import failed: {type(exc).__name__}: {exc}"
            ) from exc
    except Exception as exc:
        raise TensorRTInferenceError(
            f"TensorRT import failed: {type(exc).__name__}: {exc}"
        ) from exc


def _import_cuda_runtime() -> Any:
    try:
        from cuda.bindings import runtime as cudart

        return cudart
    except ModuleNotFoundError as exc:
        raise TensorRTInferenceError(
            "CUDA Python bindings are unavailable; install cuda-python"
        ) from exc


def _cuda_result(cudart: Any, operation: str, result: Any) -> Any:
    values = result if isinstance(result, tuple) else (result,)
    error = values[0]
    if error != cudart.cudaError_t.cudaSuccess:
        try:
            _name_error, error_name = cudart.cudaGetErrorName(error)
            if isinstance(error_name, bytes):
                error_name = error_name.decode("utf-8", errors="replace")
        except Exception:
            error_name = str(error)
        raise TensorRTInferenceError(f"{operation} failed: {error_name}")
    if len(values) == 1:
        return None
    if len(values) == 2:
        return values[1]
    return values[1:]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gpu_identity(cudart: Any) -> dict[str, Any]:
    count = int(_cuda_result(cudart, "cudaGetDeviceCount", cudart.cudaGetDeviceCount()))
    if count < 1:
        raise TensorRTInferenceError("CUDA reported no available GPU")
    properties = _cuda_result(
        cudart,
        "cudaGetDeviceProperties",
        cudart.cudaGetDeviceProperties(0),
    )
    raw_name = properties.name
    if isinstance(raw_name, bytes):
        name = raw_name.split(b"\0", 1)[0].decode("utf-8", errors="replace")
    else:
        name = str(raw_name).split("\0", 1)[0]
    runtime_version = int(
        _cuda_result(
            cudart,
            "cudaRuntimeGetVersion",
            cudart.cudaRuntimeGetVersion(),
        )
    )
    driver_version = int(
        _cuda_result(
            cudart,
            "cudaDriverGetVersion",
            cudart.cudaDriverGetVersion(),
        )
    )
    return {
        "device_index": 0,
        "name": name,
        "compute_capability": [int(properties.major), int(properties.minor)],
        "total_memory_bytes": int(properties.totalGlobalMem),
        "cuda_runtime_version": runtime_version,
        "cuda_driver_version": driver_version,
    }


class _PinnedBuffer:
    def __init__(self, cudart: Any, shape: tuple[int, ...]) -> None:
        self.cudart = cudart
        self.shape = shape
        self.size = int(np.prod(shape))
        self.nbytes = self.size * np.dtype(np.float32).itemsize
        self.host_pointer = _cuda_result(
            cudart,
            "cudaHostAlloc",
            cudart.cudaHostAlloc(self.nbytes, cudart.cudaHostAllocDefault),
        )
        host_array = (ctypes.c_float * self.size).from_address(
            int(self.host_pointer)
        )
        self.host = np.ctypeslib.as_array(host_array).reshape(shape)
        self.device_pointer = _cuda_result(
            cudart,
            "cudaMalloc",
            cudart.cudaMalloc(self.nbytes),
        )
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        _cuda_result(
            self.cudart,
            "cudaFree",
            self.cudart.cudaFree(self.device_pointer),
        )
        _cuda_result(
            self.cudart,
            "cudaFreeHost",
            self.cudart.cudaFreeHost(self.host_pointer),
        )


class _TensorRTRuntime:
    def __init__(self, engine_path: Path, trt: Any, cudart: Any) -> None:
        self.trt = trt
        self.cudart = cudart
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(
            Path(engine_path).read_bytes()
        )
        if self.engine is None:
            raise TensorRTInferenceError(f"Could not deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise TensorRTInferenceError("Could not create TensorRT execution context")
        self.stream = _cuda_result(
            cudart,
            "cudaStreamCreate",
            cudart.cudaStreamCreate(),
        )
        self.input = _PinnedBuffer(cudart, (1, *INPUT_SHAPE_CHW))
        self.output = _PinnedBuffer(cudart, (1,))
        self.input_name, self.output_name = self._resolve_tensor_names()
        self.context.set_tensor_address(
            self.input_name,
            int(self.input.device_pointer),
        )
        self.context.set_tensor_address(
            self.output_name,
            int(self.output.device_pointer),
        )
        self.closed = False

    @property
    def allocation_bytes(self) -> int:
        return self.input.nbytes + self.output.nbytes

    def _resolve_tensor_names(self) -> tuple[str, str]:
        inputs: list[str] = []
        outputs: list[str] = []
        for index in range(int(self.engine.num_io_tensors)):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(int(value) for value in self.engine.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            if dtype != self.trt.float32:
                raise TensorRTInferenceError(
                    f"TensorRT tensor {name} must use float32 I/O, found {dtype}"
                )
            if mode == self.trt.TensorIOMode.INPUT:
                inputs.append(name)
                if shape != (1, *INPUT_SHAPE_CHW):
                    raise TensorRTInferenceError(
                        f"Unexpected TensorRT input shape for {name}: {shape}"
                    )
            else:
                outputs.append(name)
                if int(np.prod(shape)) != 1:
                    raise TensorRTInferenceError(
                        f"Unexpected TensorRT output shape for {name}: {shape}"
                    )
        if len(inputs) != 1 or len(outputs) != 1:
            raise TensorRTInferenceError(
                f"Expected one TensorRT input/output, found {inputs}/{outputs}"
            )
        return inputs[0], outputs[0]

    def infer_logit(self, normalized_window: np.ndarray) -> float:
        normalized = np.asarray(normalized_window, dtype=np.float32)
        if normalized.shape != INPUT_SHAPE_CHW or not normalized.flags.c_contiguous:
            normalized = np.ascontiguousarray(normalized, dtype=np.float32)
        if normalized.shape != INPUT_SHAPE_CHW or not np.isfinite(normalized).all():
            raise ValueError(f"Invalid normalized TensorRT input: {normalized.shape}")
        np.copyto(self.input.host[0], normalized)
        _cuda_result(
            self.cudart,
            "cudaMemcpyAsync(H2D)",
            self.cudart.cudaMemcpyAsync(
                self.input.device_pointer,
                self.input.host_pointer,
                self.input.nbytes,
                self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream,
            ),
        )
        if not self.context.execute_async_v3(stream_handle=int(self.stream)):
            raise TensorRTInferenceError("TensorRT execute_async_v3 returned false")
        _cuda_result(
            self.cudart,
            "cudaMemcpyAsync(D2H)",
            self.cudart.cudaMemcpyAsync(
                self.output.host_pointer,
                self.output.device_pointer,
                self.output.nbytes,
                self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            ),
        )
        _cuda_result(
            self.cudart,
            "cudaStreamSynchronize",
            self.cudart.cudaStreamSynchronize(self.stream),
        )
        return float(self.output.host[0])

    def benchmark(
        self,
        *,
        warmup_runs: int = DEFAULT_WARMUP_RUNS,
        measured_runs: int = DEFAULT_BENCHMARK_RUNS,
    ) -> TensorRTBenchmark:
        sample = np.zeros(INPUT_SHAPE_CHW, dtype=np.float32)
        for _ in range(max(int(warmup_runs), 0)):
            self.infer_logit(sample)
        timings_ms: list[float] = []
        for _ in range(max(int(measured_runs), 1)):
            started = time.perf_counter()
            self.infer_logit(sample)
            timings_ms.append((time.perf_counter() - started) * 1e3)
        values = np.asarray(timings_ms, dtype=np.float64)
        return TensorRTBenchmark(
            warmup_runs=max(int(warmup_runs), 0),
            measured_runs=max(int(measured_runs), 1),
            p50_ms=float(np.percentile(values, 50)),
            p95_ms=float(np.percentile(values, 95)),
            max_ms=float(np.max(values)),
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.input.close()
        self.output.close()
        _cuda_result(
            self.cudart,
            "cudaStreamDestroy",
            self.cudart.cudaStreamDestroy(self.stream),
        )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class TensorRTDroneBirdInference:
    """Stateful TensorRT FP16 inference from training-exported ONNX artifacts."""

    def __init__(
        self,
        artifact_dir: Path,
        config: RadarInferenceConfig,
        profile_path: Path,
        *,
        parity_data_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.profile_path = Path(profile_path)
        self.config = config
        self._report_progress = progress_callback or (lambda _message: None)
        self._history: deque[np.ndarray] = deque(maxlen=WINDOW_STEPS)
        self._trt = _import_tensorrt()
        self._cudart = _import_cuda_runtime()
        self.gpu = _gpu_identity(self._cudart)
        self.onnx_path = self.artifact_dir / ONNX_MODEL_NAME
        self.parity_data_path = (
            Path(parity_data_path)
            if parity_data_path is not None
            else self.artifact_dir / PARITY_DATA_NAME
        )
        self.cache_dir = self.artifact_dir / "generated"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.engine_path = self.cache_dir / "model_sm87_fp16.engine"
        self.metadata_path = self.engine_path.with_suffix(".engine.json")
        self._load_deployment_artifacts()
        expected = self._expected_metadata()
        if not self._cache_is_valid(expected):
            self._report_progress(
                "TensorRT engine cache is missing or stale; compiling FP16 "
                f"engine from {self.onnx_path}. This can take several minutes."
            )
            self._build_engine(expected)
        else:
            self._report_progress(
                f"Using cached TensorRT FP16 engine: {self.engine_path}"
            )
        self.runtime = _TensorRTRuntime(
            self.engine_path,
            self._trt,
            self._cudart,
        )
        if not self._cache_is_valid(expected):
            self.close()
            raise TensorRTInferenceError("TensorRT engine cache validation failed")
        self.benchmark = self.runtime.benchmark()

    @property
    def valid_steps(self) -> int:
        return len(self._history)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "model": "drone_bird_cnn",
            "backend": "tensorrt",
            "device": "cuda",
            "precision": ENGINE_PRECISION,
            "precision_policy": ENGINE_PRECISION_POLICY,
            "engine_path": str(self.engine_path),
            "gpu": self.gpu,
            "device_allocation_bytes": self.runtime.allocation_bytes,
            "benchmark": {
                "warmup_runs": self.benchmark.warmup_runs,
                "measured_runs": self.benchmark.measured_runs,
                "p50_ms": self.benchmark.p50_ms,
                "p95_ms": self.benchmark.p95_ms,
                "max_ms": self.benchmark.max_ms,
            },
            "threshold": self.threshold,
            "labels": ["drone", "not_drone", "unknown"],
            "deployment_status": self.model_card.get("deployment_status"),
        }

    def reset(self, reason: str) -> InferenceResult:
        self._history.clear()
        return self.unknown(reason)

    def unknown(self, reason: str) -> InferenceResult:
        return InferenceResult(
            label="unknown",
            p_drone=None,
            threshold=self.threshold,
            status="waiting",
            reason=reason,
            valid_steps=self.valid_steps,
        )

    def update_feature_step(self, step: np.ndarray) -> InferenceResult:
        step = np.asarray(step, dtype=np.float32)
        if step.shape != (2, DOPPLER_BINS) or not np.isfinite(step).all():
            return self.reset(f"invalid_feature_step:{step.shape}")
        self._history.append(step)
        if len(self._history) < WINDOW_STEPS:
            return self.unknown("insufficient_history")
        try:
            window = np.stack(tuple(self._history), axis=1)
            normalized = self.normalize_feature_window(window)
            logit = self.runtime.infer_logit(normalized)
            probability = self.calibrate_logit(logit)
        except Exception as exc:
            return self.reset(f"inference_error:{type(exc).__name__}")
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            return self.reset("invalid_calibrated_probability")
        return InferenceResult(
            label="drone" if probability >= self.threshold else "not_drone",
            p_drone=probability,
            threshold=self.threshold,
            status="ready",
            reason=None,
            valid_steps=self.valid_steps,
        )

    def update(
        self,
        doppler_cube: np.ndarray,
        target_range_bin: int,
    ) -> InferenceResult:
        expected_shape = (
            int(self.config.num_loops or 0),
            int(self.config.num_chirps_per_loop or 0),
            int(self.config.num_rx_channels),
            int(self.config.num_adc_samples),
        )
        if tuple(doppler_cube.shape) != expected_shape:
            return self.reset(
                "invalid_doppler_shape:"
                f"expected={expected_shape},actual={tuple(doppler_cube.shape)}"
            )
        try:
            step = doppler_cube_to_feature_step(
                doppler_cube,
                target_range_bin,
            )
        except (TypeError, ValueError) as exc:
            return self.reset(f"invalid_feature_step:{exc}")
        return self.update_feature_step(step)

    def normalize_feature_window(self, window: np.ndarray) -> np.ndarray:
        window = np.asarray(window, dtype=np.float32)
        if window.shape != INPUT_SHAPE_CHW or not np.isfinite(window).all():
            raise ValueError(f"Invalid feature window: {window.shape}")
        normalized = (
            np.clip(
                window,
                self.clip_low[:, None, None],
                self.clip_high[:, None, None],
            )
            - self.channel_mean[:, None, None]
        ) / self.channel_std[:, None, None]
        return np.ascontiguousarray(normalized, dtype=np.float32)

    def calibrate_logit(self, logit: float) -> float:
        return float(
            self.calibrator.predict_proba(
                np.asarray([[float(logit)]], dtype=np.float64)
            )[0, 1]
        )

    def close(self) -> None:
        runtime = getattr(self, "runtime", None)
        if runtime is not None:
            runtime.close()

    @staticmethod
    def _channel_vector(calibration: dict[str, Any], key: str) -> np.ndarray:
        value = np.asarray(calibration[key], dtype=np.float32)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise TensorRTInferenceError(
                f"Classification {key} must contain two finite values"
            )
        return value

    def _load_deployment_artifacts(self) -> None:
        try:
            import joblib
        except ModuleNotFoundError as exc:
            raise TensorRTInferenceError(
                "TensorRT classification requires joblib and scikit-learn"
            ) from exc

        calibration_path = self.artifact_dir / "calibration.joblib"
        card_path = self.artifact_dir / "manifest.json"
        for path in (
            self.onnx_path,
            self.parity_data_path,
            calibration_path,
            card_path,
        ):
            if not path.is_file():
                raise TensorRTInferenceError(
                    f"TensorRT classification artifact not found: {path}"
                )
        try:
            calibration = joblib.load(calibration_path)
            card = json.loads(card_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise TensorRTInferenceError(
                f"Could not load TensorRT deployment artifacts: {type(exc).__name__}"
            ) from exc
        required = {
            "calibrator",
            "threshold",
            "clip_low",
            "clip_high",
            "channel_mean",
            "channel_std",
            "compatible_profile_sha256",
        }
        if not isinstance(calibration, dict) or required.difference(calibration):
            missing = required.difference(calibration) if isinstance(calibration, dict) else required
            raise TensorRTInferenceError(
                "CNN calibration artifact is missing: " + ", ".join(sorted(missing))
            )
        self.threshold = float(calibration["threshold"])
        if not np.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise TensorRTInferenceError("CNN threshold must be a finite probability")
        self.clip_low = self._channel_vector(calibration, "clip_low")
        self.clip_high = self._channel_vector(calibration, "clip_high")
        self.channel_mean = self._channel_vector(calibration, "channel_mean")
        self.channel_std = self._channel_vector(calibration, "channel_std")
        if np.any(self.clip_low > self.clip_high) or np.any(self.channel_std <= 0.0):
            raise TensorRTInferenceError("CNN normalization values are invalid")
        self.calibrator = calibration["calibrator"]
        self.model_card = card

        profile_hash = normalized_profile_sha256(self.profile_path)
        if calibration["compatible_profile_sha256"] != profile_hash:
            raise TensorRTInferenceError("Radar configuration does not match the CNN")
        if not isinstance(card, dict):
            raise TensorRTInferenceError("CNN model card must be a mapping")
        if tuple(card.get("input_shape_chw", ())) != INPUT_SHAPE_CHW:
            raise TensorRTInferenceError("CNN model card input shape is incompatible")
        if card.get("feature_version") != FEATURE_VERSION:
            raise TensorRTInferenceError("CNN model card feature version is incompatible")
        if int(card.get("target_gate_range_bins", -1)) != TARGET_GATE_BINS:
            raise TensorRTInferenceError("CNN target range-bin contract is incompatible")
        if card.get("compatible_profile_sha256") != profile_hash:
            raise TensorRTInferenceError("CNN model card profile fingerprint is incompatible")
        if card.get("onnx_sha256") != _sha256(self.onnx_path):
            raise TensorRTInferenceError("CNN ONNX hash does not match the model card")
        if card.get("parity_data_sha256") != _sha256(self.parity_data_path):
            raise TensorRTInferenceError(
                "CNN parity-data hash does not match the model card"
            )

        expected_dimensions = (64, 4, 384, 128, 3)
        observed_dimensions = (
            int(self.config.num_adc_samples),
            int(self.config.num_rx_channels),
            int(self.config.num_chirps_per_frame),
            int(self.config.num_loops or 0),
            int(self.config.num_chirps_per_loop or 0),
        )
        if observed_dimensions != expected_dimensions:
            raise TensorRTInferenceError(
                "Radar dimensions are incompatible with the CNN: "
                f"expected={expected_dimensions}, observed={observed_dimensions}"
            )
        if tuple(self.config.tx_channel_masks or ()) != (1, 4, 2):
            raise TensorRTInferenceError("Radar TX schedule is incompatible with the CNN")

    def _expected_metadata(self) -> dict[str, Any]:
        calibration = self.artifact_dir / "calibration.joblib"
        card = self.artifact_dir / "manifest.json"
        return {
            "format_version": ENGINE_FORMAT_VERSION,
            "precision": ENGINE_PRECISION,
            "input_shape": [1, *INPUT_SHAPE_CHW],
            "workspace_bytes": ENGINE_WORKSPACE_BYTES,
            "precision_policy": ENGINE_PRECISION_POLICY,
            "onnx_sha256": _sha256(self.onnx_path),
            "parity_data_sha256": _sha256(self.parity_data_path),
            "calibration_sha256": _sha256(calibration),
            "model_card_sha256": _sha256(card),
            "profile_sha256": _sha256(self.profile_path),
            "tensorrt_version": str(self._trt.__version__),
            "gpu": self.gpu,
        }

    def _cache_is_valid(self, expected: dict[str, Any]) -> bool:
        if not self.engine_path.is_file() or not self.metadata_path.is_file():
            return False
        try:
            observed = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        for key, value in expected.items():
            if observed.get(key) != value:
                return False
        if observed.get("engine_sha256") != _sha256(self.engine_path):
            return False
        parity = observed.get("parity")
        return bool(
            isinstance(parity, dict)
            and parity.get("label_mismatches") == 0
            and float(parity.get("max_probability_error", float("inf")))
            <= PARITY_PROBABILITY_TOLERANCE
        )

    def _build_engine(
        self,
        expected: dict[str, Any],
    ) -> None:
        logger = self._trt.Logger(self._trt.Logger.WARNING)
        builder = self._trt.Builder(logger)
        network_flags = 1 << int(
            self._trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
        )
        network = builder.create_network(network_flags)
        parser = self._trt.OnnxParser(network, logger)
        if not parser.parse(self.onnx_path.read_bytes()):
            errors = [
                str(parser.get_error(index))
                for index in range(int(parser.num_errors))
            ]
            raise TensorRTInferenceError(
                "TensorRT ONNX parsing failed: " + "; ".join(errors)
            )
        build_config = builder.create_builder_config()
        if not bool(builder.platform_has_fast_fp16):
            raise TensorRTInferenceError(
                "TensorRT reports that this GPU has no fast FP16 support"
            )
        build_config.set_memory_pool_limit(
            self._trt.MemoryPoolType.WORKSPACE,
            ENGINE_WORKSPACE_BYTES,
        )
        build_config.set_flag(self._trt.BuilderFlag.FP16)
        self._report_progress(
            "TensorRT ONNX parsing complete; compiling optimized FP16 engine..."
        )
        serialized = builder.build_serialized_network(network, build_config)
        if serialized is None:
            raise TensorRTInferenceError("TensorRT engine build returned no engine")
        temporary_engine = self.engine_path.with_suffix(".engine.tmp")
        temporary_engine.write_bytes(bytes(serialized))
        os.replace(temporary_engine, self.engine_path)
        self._report_progress(
            f"TensorRT engine compiled: {self.engine_path}"
        )

        self._report_progress(
            f"Validating TensorRT engine with parity data: {self.parity_data_path}"
        )
        runtime = _TensorRTRuntime(
            self.engine_path,
            self._trt,
            self._cudart,
        )
        try:
            parity = self._validate_parity(runtime)
        finally:
            runtime.close()
        if (
            parity["label_mismatches"] != 0
            or parity["max_probability_error"] > PARITY_PROBABILITY_TOLERANCE
        ):
            self.engine_path.unlink(missing_ok=True)
            raise TensorRTInferenceError(
                "TensorRT FP16 parity validation failed: "
                f"label_mismatches={parity['label_mismatches']}, "
                f"max_probability_error={parity['max_probability_error']:.6g}"
            )
        metadata = {
            **expected,
            "engine_sha256": _sha256(self.engine_path),
            "parity": parity,
        }
        temporary_metadata = self.metadata_path.with_suffix(".json.tmp")
        temporary_metadata.write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_metadata, self.metadata_path)
        self._report_progress(
            "TensorRT parity validation passed: "
            f"windows={parity['windows']}, "
            f"max_probability_error={parity['max_probability_error']:.6g}. "
            f"Engine cache ready: {self.engine_path}"
        )

    def _validate_parity(
        self,
        runtime: _TensorRTRuntime,
    ) -> dict[str, Any]:
        with np.load(self.parity_data_path, allow_pickle=False) as parity_data:
            normalized = np.asarray(
                parity_data["normalized_windows"], dtype=np.float32
            )
            expected_probabilities = np.asarray(
                parity_data["probabilities"], dtype=np.float64
            )
        if (
            normalized.ndim != 4
            or len(normalized) == 0
            or tuple(normalized.shape[1:]) != INPUT_SHAPE_CHW
        ):
            raise TensorRTInferenceError(
                f"Unexpected parity input shape: {normalized.shape}"
            )
        if expected_probabilities.shape != (len(normalized),):
            raise TensorRTInferenceError(
                "TensorRT parity probabilities do not match the input count"
            )
        if (
            not np.isfinite(normalized).all()
            or not np.isfinite(expected_probabilities).all()
            or np.any((expected_probabilities < 0.0) | (expected_probabilities > 1.0))
        ):
            raise TensorRTInferenceError("TensorRT parity data contains non-finite values")
        normalized = np.ascontiguousarray(normalized, dtype=np.float32)
        gpu_logits = np.asarray(
            [runtime.infer_logit(window) for window in normalized],
            dtype=np.float64,
        )
        gpu_probabilities = self.calibrator.predict_proba(
            gpu_logits[:, np.newaxis]
        )[:, 1]
        maximum_error = float(
            np.max(np.abs(expected_probabilities - gpu_probabilities))
        )
        label_mismatches = int(
            np.count_nonzero(
                (expected_probabilities >= self.threshold)
                != (gpu_probabilities >= self.threshold)
            )
        )
        return {
            "windows": int(normalized.shape[0]),
            "label_mismatches": int(label_mismatches),
            "max_probability_error": float(maximum_error),
            "tolerance": PARITY_PROBABILITY_TOLERANCE,
        }

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
