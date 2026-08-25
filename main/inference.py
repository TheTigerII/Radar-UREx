"""Real-time mmHawkeye LSTM inference for Mini4 Doppler-Time histories."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


SEGMENT_FRAMES = 36
DOPPLER_BINS = 64
CENTER_BIN = DOPPLER_BINS // 2
CENTER_TOLERANCE_BINS = 1
LABEL_TO_INDEX = {"other": 0, "uav": 1}
DEFAULT_MODEL_NAME = "model.onnx"
DEFAULT_TENSORRT_ENGINE_NAME = "model.fp16.engine"
GENERATED_ENGINE_DIRECTORY_NAME = "generated"
DEFAULT_MANIFEST_NAME = "manifest.json"


def align_doppler_time(history_db: np.ndarray) -> np.ndarray:
    """Apply paper-style DC removal, body-peak alignment, and compression."""
    history = np.asarray(history_db, dtype=np.float32)
    if history.shape != (SEGMENT_FRAMES, DOPPLER_BINS):
        raise ValueError(
            f"Expected [{SEGMENT_FRAMES}, {DOPPLER_BINS}], got {history.shape}"
        )
    if not np.isfinite(history).all():
        raise ValueError("Doppler-Time history contains non-finite values")
    amplitude = np.power(10.0, history / 20.0, dtype=np.float32)
    body_peak_bins = np.argmax(amplitude, axis=1)
    away_from_dc = np.abs(body_peak_bins - CENTER_BIN) > CENTER_TOLERANCE_BINS
    dc_removed = amplitude.copy()
    if np.any(away_from_dc):
        dc_noise = float(np.mean(amplitude[away_from_dc, CENTER_BIN]))
        dc_removed[:, CENTER_BIN] = np.maximum(
            dc_removed[:, CENTER_BIN] - dc_noise,
            0.0,
        )
    coordinates = np.arange(DOPPLER_BINS, dtype=np.float32)
    aligned = np.empty_like(amplitude)
    for time_index, spectrum in enumerate(dc_removed):
        body_peak_bin = int(body_peak_bins[time_index])
        if abs(body_peak_bin - CENTER_BIN) <= CENTER_TOLERANCE_BINS:
            aligned[time_index] = spectrum
            continue
        shift = CENTER_BIN - body_peak_bin
        aligned[time_index] = np.interp(
            coordinates - shift,
            coordinates,
            spectrum,
            left=0.0,
            right=0.0,
        ).astype(np.float32)
    return np.log1p(aligned).astype(np.float32)


def resolve_model_artifact_paths(model_weights_dir: Path) -> tuple[Path, Path]:
    directory = Path(model_weights_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Model weights directory does not exist: {directory}")
    model_path = directory / DEFAULT_MODEL_NAME
    manifest_path = directory / DEFAULT_MANIFEST_NAME
    missing = [path for path in (model_path, manifest_path) if not path.is_file()]
    if missing:
        expected = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"ONNX model artifacts are missing: expected {expected}")
    return model_path, manifest_path


def _load_tensorrt() -> tuple[Any, Any]:
    try:
        import tensorrt as trt
        from cuda.bindings import runtime as cudart
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT and CUDA Python bindings are required to use the "
            f"{DEFAULT_TENSORRT_ENGINE_NAME} classifier engine"
        ) from exc
    return trt, cudart


def ensure_tensorrt_engine(model_path: Path) -> Path:
    """Return the generated TensorRT engine, building it with visible output."""
    generated_directory = model_path.parent / GENERATED_ENGINE_DIRECTORY_NAME
    engine_path = generated_directory / DEFAULT_TENSORRT_ENGINE_NAME
    if engine_path.is_file():
        return engine_path
    existing_engines = (
        sorted(generated_directory.glob("*.engine"))
        if generated_directory.is_dir()
        else []
    )
    if len(existing_engines) == 1:
        return existing_engines[0]
    if len(existing_engines) > 1:
        paths = ", ".join(str(path) for path in existing_engines)
        raise RuntimeError(
            "Multiple TensorRT engines were found without the preferred "
            f"{engine_path.name}: {paths}"
        )

    trtexec = shutil.which("trtexec")
    if trtexec is None:
        raise RuntimeError(
            "TensorRT engine is missing and trtexec was not found in PATH"
        )

    generated_directory.mkdir(parents=True, exist_ok=True)
    building_path = engine_path.with_name(engine_path.name + ".building")
    if building_path.exists():
        building_path.unlink()
    command = [
        trtexec,
        f"--onnx={model_path}",
        f"--saveEngine={building_path}",
        "--fp16",
        f"--shapes=doppler_time:1x{SEGMENT_FRAMES}x{DOPPLER_BINS}",
        "--skipInference",
    ]
    print(
        "TensorRT engine not found; compiling it now on the Jetson GPU.",
        flush=True,
    )
    print(f"TensorRT engine destination: {engine_path}", flush=True)
    print("trtexec output follows:", flush=True)
    try:
        completed = subprocess.run(command, check=False)
    except OSError as exc:
        raise RuntimeError(f"Could not start trtexec: {exc}") from exc
    if completed.returncode != 0:
        building_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"TensorRT engine compilation failed with exit code "
            f"{completed.returncode}"
        )
    if not building_path.is_file() or building_path.stat().st_size == 0:
        building_path.unlink(missing_ok=True)
        raise RuntimeError("trtexec completed without creating a TensorRT engine")
    building_path.replace(engine_path)
    print(f"TensorRT engine compiled successfully: {engine_path}", flush=True)
    return engine_path


class _TensorRtSession:
    """Small synchronous TensorRT adapter with persistent device buffers."""

    def __init__(self, engine_path: Path) -> None:
        self.trt, self.cudart = _load_tensorrt()
        self.logger = self.trt.Logger(self.trt.Logger.WARNING)
        self.runtime = self.trt.Runtime(self.logger)
        engine_bytes = engine_path.read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Could not create TensorRT execution context")
        self._validate_interface()
        self.input = np.empty(
            (1, SEGMENT_FRAMES, DOPPLER_BINS),
            dtype=np.float32,
        )
        self.output = np.empty((1, len(LABEL_TO_INDEX)), dtype=np.float32)
        self.stream = self._cuda_value(
            self.cudart.cudaStreamCreate(),
            "create TensorRT CUDA stream",
        )
        self.input_device = self._cuda_value(
            self.cudart.cudaMalloc(self.input.nbytes),
            "allocate TensorRT input buffer",
        )
        try:
            self.output_device = self._cuda_value(
                self.cudart.cudaMalloc(self.output.nbytes),
                "allocate TensorRT output buffer",
            )
        except Exception:
            self.cudart.cudaFree(self.input_device)
            raise
        if not self.context.set_tensor_address(
            "doppler_time", int(self.input_device)
        ) or not self.context.set_tensor_address("logits", int(self.output_device)):
            self.close()
            raise RuntimeError("Could not bind TensorRT input/output buffers")

    def _cuda_value(self, result: tuple[Any, ...], operation: str) -> Any:
        error, *values = result
        if error != self.cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA failed to {operation}: {error}")
        return values[0] if values else None

    def _validate_interface(self) -> None:
        names = {
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
        }
        if names != {"doppler_time", "logits"}:
            raise ValueError(
                "TensorRT engine must expose doppler_time and logits tensors"
            )
        expected = {
            "doppler_time": (1, SEGMENT_FRAMES, DOPPLER_BINS),
            "logits": (1, len(LABEL_TO_INDEX)),
        }
        for name, shape in expected.items():
            if tuple(self.engine.get_tensor_shape(name)) != shape:
                raise ValueError(
                    f"TensorRT {name} shape must be {shape}, got "
                    f"{tuple(self.engine.get_tensor_shape(name))}"
                )
            if self.engine.get_tensor_dtype(name) != self.trt.float32:
                raise ValueError(f"TensorRT {name} tensor must use float32 I/O")

    def run(
        self,
        output_names: list[str],
        inputs: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        if output_names != ["logits"] or set(inputs) != {"doppler_time"}:
            raise ValueError("Unexpected TensorRT input or output names")
        values = np.asarray(inputs["doppler_time"], dtype=np.float32)
        if values.shape != self.input.shape:
            raise ValueError(
                f"TensorRT input must have shape {self.input.shape}, got {values.shape}"
            )
        np.copyto(self.input, values)
        self._cuda_value(
            self.cudart.cudaMemcpyAsync(
                int(self.input_device),
                self.input.ctypes.data,
                self.input.nbytes,
                self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream,
            ),
            "copy classifier input to the GPU",
        )
        if not self.context.execute_async_v3(int(self.stream)):
            raise RuntimeError("TensorRT inference execution failed")
        self._cuda_value(
            self.cudart.cudaMemcpyAsync(
                self.output.ctypes.data,
                int(self.output_device),
                self.output.nbytes,
                self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            ),
            "copy classifier output from the GPU",
        )
        self._cuda_value(
            self.cudart.cudaStreamSynchronize(self.stream),
            "synchronize classifier inference",
        )
        return [self.output.copy()]

    def close(self) -> None:
        output_device = getattr(self, "output_device", None)
        input_device = getattr(self, "input_device", None)
        stream = getattr(self, "stream", None)
        if stream is not None:
            self.cudart.cudaStreamSynchronize(stream)
        if output_device is not None:
            self.cudart.cudaFree(output_device)
            self.output_device = None
        if input_device is not None:
            self.cudart.cudaFree(input_device)
            self.input_device = None
        if stream is not None:
            self.cudart.cudaStreamDestroy(stream)
            self.stream = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


@dataclass(frozen=True)
class ClassificationResult:
    status: str
    label: str
    history_frames: int
    maximum_pmm_score: Optional[float]
    threshold: float
    probabilities: Optional[dict[str, float]] = None
    confidence: Optional[float] = None
    inference_ms: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RealtimeUavClassifier:
    """Stateful rolling classifier with PMM quality and contract checks."""

    def __init__(
        self,
        model_weights_dir: Path,
        runtime_contract: dict[str, Any],
        *,
        device: Optional[str] = None,
    ) -> None:
        self.model_path, self.manifest_path = resolve_model_artifact_paths(
            model_weights_dir
        )
        requested_device = device.lower() if device is not None else None
        if requested_device not in {None, "cuda", "cuda:0", "tensorrt"}:
            raise ValueError("Live classification is TensorRT GPU-only")
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Model manifest is not valid JSON: {self.manifest_path}"
            ) from exc
        if not isinstance(manifest, dict):
            raise ValueError("Model manifest must contain a metadata object")
        self._validate_manifest(manifest, runtime_contract)
        external_data_name = manifest["artifacts"].get("onnx_data")
        self.external_data_path: Optional[Path] = None
        if external_data_name is not None:
            self.external_data_path = self.model_path.parent / external_data_name
            if not self.external_data_path.is_file():
                raise FileNotFoundError(
                    "ONNX external data is missing: expected "
                    f"{self.external_data_path}"
                )

        self.engine_path = ensure_tensorrt_engine(self.model_path)
        self.session = _TensorRtSession(self.engine_path)
        self.runtime_name = "tensorrt"
        self.providers = ["TensorRT"]
        self.device = "cuda"

        self.mean = float(manifest["normalization"]["mean"])
        self.std = float(manifest["normalization"]["std"])
        dataset = manifest["dataset"]
        self.profile_fingerprint = str(dataset["profile_fingerprint"])
        self.feature_fingerprint = str(dataset["feature_fingerprint"])
        self.feature_version = str(dataset["feature_version"])
        self.score_history: deque[float] = deque(maxlen=SEGMENT_FRAMES)
        self.pmm_gate_history: deque[bool] = deque(maxlen=SEGMENT_FRAMES)
        self.previous_history_frames = 0

    @staticmethod
    def _validate_manifest(
        manifest: dict[str, Any],
        runtime_contract: dict[str, Any],
    ) -> None:
        required = {
            "artifacts",
            "architecture",
            "input_contract",
            "label_to_index",
            "normalization",
            "dataset",
        }
        missing = sorted(required - manifest.keys())
        if missing:
            raise ValueError(f"Model manifest is missing fields: {missing}")
        artifacts = manifest["artifacts"]
        if (
            not isinstance(artifacts, dict)
            or artifacts.get("onnx") != DEFAULT_MODEL_NAME
        ):
            raise ValueError(f"Model manifest must reference {DEFAULT_MODEL_NAME}")
        external_data_name = artifacts.get("onnx_data")
        if external_data_name is not None and (
            not isinstance(external_data_name, str)
            or Path(external_data_name).name != external_data_name
        ):
            raise ValueError("Model manifest has an invalid ONNX external data name")
        input_contract = manifest["input_contract"]
        expected_shape = ["batch", SEGMENT_FRAMES, DOPPLER_BINS]
        if (
            not isinstance(input_contract, dict)
            or input_contract.get("dtype") != "float32"
            or input_contract.get("shape") != expected_shape
        ):
            raise ValueError("ONNX input contract is not compatible with Mini4 live data")
        if manifest["label_to_index"] != LABEL_TO_INDEX:
            raise ValueError("Model label order must be other=0 and uav=1")
        architecture = manifest["architecture"]
        expected_architecture = {
            "input_size": DOPPLER_BINS,
            "hidden_size": 128,
            "num_layers": 2,
            "num_classes": 2,
        }
        if architecture != expected_architecture:
            raise ValueError(
                "Model architecture is incompatible: "
                f"expected={expected_architecture}, observed={architecture}"
            )
        normalization = manifest["normalization"]
        try:
            mean = float(normalization["mean"])
            std = float(normalization["std"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Model normalization metadata is invalid") from exc
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0.0:
            raise ValueError("Model normalization values must be finite and non-zero")
        dataset = manifest["dataset"]
        if not isinstance(dataset, dict):
            raise ValueError("Model dataset metadata must be an object")
        for field in (
            "feature_version",
            "feature_fingerprint",
            "profile_fingerprint",
        ):
            observed = runtime_contract.get(field)
            expected = dataset.get(field)
            if observed != expected:
                raise ValueError(
                    f"Model {field} mismatch: expected={expected}, observed={observed}"
                )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "runtime": self.runtime_name,
            "model_path": str(self.engine_path),
            "onnx_model_path": str(self.model_path),
            "manifest_path": str(self.manifest_path),
            "external_data_path": (
                str(self.external_data_path)
                if self.external_data_path is not None
                else None
            ),
            "input_shape": [SEGMENT_FRAMES, DOPPLER_BINS],
            "label_to_index": dict(LABEL_TO_INDEX),
            "profile_fingerprint": self.profile_fingerprint,
            "feature_fingerprint": self.feature_fingerprint,
            "feature_version": self.feature_version,
            "device": self.device,
            "providers": list(self.providers),
        }

    def reset(self) -> None:
        self.score_history.clear()
        self.pmm_gate_history.clear()
        self.previous_history_frames = 0

    def classify(
        self,
        doppler_time_db: np.ndarray,
        *,
        pmm_score: Optional[float],
        threshold: float,
    ) -> ClassificationResult:
        history = np.asarray(doppler_time_db, dtype=np.float32).T
        if history.ndim != 2 or history.shape[1] != DOPPLER_BINS:
            raise ValueError(
                f"Live Doppler-Time history must have shape [T, {DOPPLER_BINS}]"
            )
        history_frames = int(history.shape[0])
        if history_frames == 0 or history_frames < self.previous_history_frames:
            self.score_history.clear()
            self.pmm_gate_history.clear()
        threshold_value = float(threshold)
        if not math.isfinite(threshold_value):
            raise ValueError("PMM threshold must be finite")
        if history_frames > 0 and pmm_score is not None:
            score = float(pmm_score)
            if not math.isfinite(score):
                raise ValueError("PMM score must be finite")
            self.score_history.append(score)
            self.pmm_gate_history.append(score >= threshold_value)
        self.previous_history_frames = history_frames
        maximum_score = (
            max(self.score_history) if self.score_history else None
        )
        if (
            history.shape != (SEGMENT_FRAMES, DOPPLER_BINS)
            or len(self.score_history) < SEGMENT_FRAMES
        ):
            return ClassificationResult(
                status="warming_up",
                label="unknown",
                history_frames=history_frames,
                maximum_pmm_score=maximum_score,
                threshold=threshold_value,
            )
        if not any(self.pmm_gate_history):
            return ClassificationResult(
                status="below_pmm_threshold",
                label="unknown",
                history_frames=history_frames,
                maximum_pmm_score=maximum_score,
                threshold=threshold_value,
            )
        features = align_doppler_time(history)
        normalized = ((features - self.mean) / self.std).astype(np.float32)
        started = time.perf_counter()
        logits = np.asarray(
            self.session.run(
                ["logits"],
                {"doppler_time": normalized[None]},
            )[0],
            dtype=np.float32,
        )
        inference_ms = (time.perf_counter() - started) * 1_000.0
        if (
            logits.shape != (1, len(LABEL_TO_INDEX))
            or not np.isfinite(logits).all()
        ):
            raise RuntimeError(
                "Classifier returned invalid logits; expected one finite two-class row"
            )
        shifted_logits = logits[0] - np.max(logits[0])
        exponentials = np.exp(shifted_logits)
        probability_values = exponentials / np.sum(exponentials)
        probabilities = {
            label: float(probability_values[index])
            for label, index in LABEL_TO_INDEX.items()
        }
        label = max(probabilities, key=probabilities.__getitem__)
        return ClassificationResult(
            status="classified",
            label=label,
            history_frames=history_frames,
            maximum_pmm_score=maximum_score,
            threshold=threshold_value,
            probabilities=probabilities,
            confidence=probabilities[label],
            inference_ms=inference_ms,
        )
