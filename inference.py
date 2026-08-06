"""Real-time mmHawkeye LSTM inference for Mini4 Doppler-Time histories."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn


SEGMENT_FRAMES = 36
DOPPLER_BINS = 64
CENTER_BIN = DOPPLER_BINS // 2
CENTER_TOLERANCE_BINS = 1
LABEL_TO_INDEX = {"other": 0, "uav": 1}
DEFAULT_MODEL_STATE_NAME = "model_state.pt"


class MmHawkeyeLSTM(nn.Module):
    """Two-layer LSTM architecture exported by training.ipynb."""

    def __init__(
        self,
        input_size: int = DOPPLER_BINS,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(self.hidden_size, int(num_classes))

    def forward(self, doppler_time: torch.Tensor) -> torch.Tensor:
        if doppler_time.ndim != 3 or tuple(doppler_time.shape[1:]) != (
            SEGMENT_FRAMES,
            DOPPLER_BINS,
        ):
            raise ValueError(
                "Expected Doppler-Time input with shape "
                f"[batch, {SEGMENT_FRAMES}, {DOPPLER_BINS}], got "
                f"{tuple(doppler_time.shape)}"
            )
        _, (hidden, _) = self.lstm(doppler_time)
        return self.classifier(hidden[-1])


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


def resolve_model_state_path(model_weights_dir: Path) -> Path:
    directory = Path(model_weights_dir).expanduser().resolve()
    path = directory / DEFAULT_MODEL_STATE_NAME
    if not directory.is_dir():
        raise FileNotFoundError(f"Model weights directory does not exist: {directory}")
    if not path.is_file():
        raise FileNotFoundError(
            f"Model weights are missing: expected {path}"
        )
    return path


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
        device: Optional[str | torch.device] = None,
    ) -> None:
        self.model_state_path = resolve_model_state_path(model_weights_dir)
        checkpoint = torch.load(
            self.model_state_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError("Model checkpoint must contain a metadata dictionary")
        self._validate_checkpoint(checkpoint, runtime_contract)
        architecture = checkpoint["architecture"]
        self.model = MmHawkeyeLSTM(**architecture)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        selected_device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device = selected_device
        self.model.to(self.device).eval()
        self.mean = float(checkpoint["normalization"]["mean"])
        self.std = float(checkpoint["normalization"]["std"])
        self.profile_fingerprint = str(checkpoint["profile_fingerprint"])
        self.feature_fingerprint = str(checkpoint["feature_fingerprint"])
        self.feature_version = str(checkpoint["feature_version"])
        self.score_history: deque[float] = deque(maxlen=SEGMENT_FRAMES)
        self.previous_history_frames = 0

    @staticmethod
    def _validate_checkpoint(
        checkpoint: dict[str, Any],
        runtime_contract: dict[str, Any],
    ) -> None:
        required = {
            "state_dict",
            "architecture",
            "input_shape",
            "label_to_index",
            "normalization",
            "profile_fingerprint",
            "feature_fingerprint",
            "feature_version",
        }
        missing = sorted(required - checkpoint.keys())
        if missing:
            raise ValueError(f"Model checkpoint is missing fields: {missing}")
        if list(checkpoint["input_shape"]) != [SEGMENT_FRAMES, DOPPLER_BINS]:
            raise ValueError("Model input shape is not compatible with Mini4 live data")
        if checkpoint["label_to_index"] != LABEL_TO_INDEX:
            raise ValueError("Model label order must be other=0 and uav=1")
        architecture = checkpoint["architecture"]
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
        normalization = checkpoint["normalization"]
        try:
            mean = float(normalization["mean"])
            std = float(normalization["std"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Model normalization metadata is invalid") from exc
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0.0:
            raise ValueError("Model normalization values must be finite and non-zero")
        for field in (
            "profile_fingerprint",
            "feature_fingerprint",
            "feature_version",
        ):
            observed = runtime_contract.get(field)
            expected = checkpoint.get(field)
            if observed != expected:
                raise ValueError(
                    f"Model {field} mismatch: expected={expected}, observed={observed}"
                )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "model_state_path": str(self.model_state_path),
            "input_shape": [SEGMENT_FRAMES, DOPPLER_BINS],
            "label_to_index": dict(LABEL_TO_INDEX),
            "profile_fingerprint": self.profile_fingerprint,
            "feature_fingerprint": self.feature_fingerprint,
            "feature_version": self.feature_version,
            "device": str(self.device),
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
        with torch.inference_mode():
            tensor = torch.from_numpy(normalized[None]).to(self.device)
            logits = self.model(tensor)
            probability_values = torch.softmax(logits, dim=1)[0].cpu().numpy()
        inference_ms = (time.perf_counter() - started) * 1_000.0
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
