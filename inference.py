"""Real-time mmHawkeye LSTM inference for Mini4 Doppler-Time histories."""

from __future__ import annotations

import json
import math
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
DEFAULT_MANIFEST_NAME = "manifest.json"


def align_doppler_time(history_db: np.ndarray) -> np.ndarray:
    """Apply the exact alignment and compression contract used for training."""
    history = np.asarray(history_db, dtype=np.float32)
    if history.shape != (SEGMENT_FRAMES, DOPPLER_BINS):
        raise ValueError(
            f"Expected [{SEGMENT_FRAMES}, {DOPPLER_BINS}], got {history.shape}"
        )
    if not np.isfinite(history).all():
        raise ValueError("Doppler-Time history contains non-finite values")
    amplitude = np.power(10.0, history / 20.0, dtype=np.float32)
    coordinates = np.arange(DOPPLER_BINS, dtype=np.float32)
    aligned = np.empty_like(amplitude)
    for time_index, spectrum in enumerate(amplitude):
        strongest_bin = int(np.argmax(spectrum))
        if abs(strongest_bin - CENTER_BIN) <= CENTER_TOLERANCE_BINS:
            aligned[time_index] = spectrum
            continue
        shift = CENTER_BIN - strongest_bin
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


def _load_onnxruntime() -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX Runtime is required for classification; install onnxruntime "
            "or an appropriate onnxruntime-gpu build"
        ) from exc
    return ort


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

        ort = _load_onnxruntime()
        providers = self._select_providers(ort, device)
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers,
        )
        self._validate_onnx_interface()
        self.providers = list(self.session.get_providers())
        self.device = "cuda" if "CUDAExecutionProvider" in self.providers else "cpu"

        self.mean = float(manifest["normalization"]["mean"])
        self.std = float(manifest["normalization"]["std"])
        dataset = manifest["dataset"]
        self.profile_fingerprint = str(dataset["profile_fingerprint"])
        self.feature_fingerprint = str(dataset["feature_fingerprint"])
        self.feature_version = str(dataset["feature_version"])
        self.score_history: deque[float] = deque(maxlen=SEGMENT_FRAMES)
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
            "profile_fingerprint",
            "feature_fingerprint",
            "feature_version",
        ):
            observed = runtime_contract.get(field)
            expected = dataset.get(field)
            if observed != expected:
                raise ValueError(
                    f"Model {field} mismatch: expected={expected}, observed={observed}"
                )

    @staticmethod
    def _select_providers(ort: Any, device: Optional[str]) -> list[str]:
        available = list(ort.get_available_providers())
        requested = device.lower() if device is not None else None
        if requested is not None and requested not in {"cpu", "cuda", "cuda:0"}:
            raise ValueError("ONNX device must be 'cpu' or 'cuda'")
        if requested in {"cuda", "cuda:0"}:
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError("ONNX Runtime CUDAExecutionProvider is not available")
            providers = ["CUDAExecutionProvider"]
            if "CPUExecutionProvider" in available:
                providers.append("CPUExecutionProvider")
            return providers
        if requested == "cpu":
            if "CPUExecutionProvider" not in available:
                raise RuntimeError("ONNX Runtime CPUExecutionProvider is not available")
            return ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider"]
            if "CPUExecutionProvider" in available:
                providers.append("CPUExecutionProvider")
            return providers
        if "CPUExecutionProvider" in available:
            return ["CPUExecutionProvider"]
        raise RuntimeError("ONNX Runtime has no supported CPU or CUDA provider")

    def _validate_onnx_interface(self) -> None:
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1:
            raise ValueError("ONNX model must have exactly one input")
        model_input = inputs[0]
        if (
            model_input.name != "doppler_time"
            or model_input.type != "tensor(float)"
            or list(model_input.shape[1:]) != [SEGMENT_FRAMES, DOPPLER_BINS]
        ):
            raise ValueError(
                "ONNX input must be float32 doppler_time with shape [batch, 36, 64]"
            )
        if len(outputs) != 1:
            raise ValueError("ONNX model must have exactly one output")
        model_output = outputs[0]
        if (
            model_output.name != "logits"
            or model_output.type != "tensor(float)"
            or list(model_output.shape[1:]) != [len(LABEL_TO_INDEX)]
        ):
            raise ValueError("ONNX output must be float32 logits with shape [batch, 2]")

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "runtime": "onnxruntime",
            "model_path": str(self.model_path),
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
        if history_frames > 0 and pmm_score is not None:
            score = float(pmm_score)
            if not math.isfinite(score):
                raise ValueError("PMM score must be finite")
            self.score_history.append(score)
        self.previous_history_frames = history_frames
        threshold_value = float(threshold)
        if not math.isfinite(threshold_value):
            raise ValueError("PMM threshold must be finite")
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
        if maximum_score is None or maximum_score < threshold_value:
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
                "ONNX model returned invalid logits; expected one finite two-class row"
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
