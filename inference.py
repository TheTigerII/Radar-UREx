from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Protocol

import numpy as np


FEATURE_VERSION = "iwr6843-micro-doppler-v4-two-range-bin"
WINDOW_STEPS = 48
DOPPLER_BINS = 64
INPUT_SHAPE_CHW = (2, WINDOW_STEPS, DOPPLER_BINS)
TARGET_GATE_BINS = 2
CNN_CONFIG = {
    "input_channels": 2,
    "stem_channels": 48,
    "block_channels": [96, 208, 416],
    "block_strides": [(1, 2), (2, 2), (2, 2)],
    "dropout": 0.25,
}
MODEL_STATE_NAME = "drone_bird_cnn_state.pt"
CALIBRATION_NAME = "drone_bird_cnn_calibration.joblib"
MODEL_CARD_NAME = "drone_bird_cnn.model_card.json"
DEPLOYMENT_STATUS = "native_iwr6843_two_range_bin_cnn"

ClassificationLabel = Literal["drone", "not_drone", "unknown"]


class RadarInferenceConfig(Protocol):
    num_adc_samples: int
    num_rx_channels: int
    num_chirps_per_frame: int
    num_loops: Optional[int]
    num_chirps_per_loop: Optional[int]
    tx_channel_masks: Optional[tuple[int, ...]]


@dataclass(frozen=True)
class InferenceResult:
    label: ClassificationLabel
    p_drone: Optional[float]
    threshold: Optional[float]
    status: str
    reason: Optional[str]
    valid_steps: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalized_profile_sha256(profile_path: Path) -> str:
    """Hash profile text with LF endings, matching the Linux training run."""
    profile_bytes = Path(profile_path).read_bytes()
    normalized = profile_bytes.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def _range_indices(center: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    # Use the selected bin and its immediately nearer neighbour. This must stay
    # deterministic so offline feature generation and live inference agree.
    target = np.arange(max(0, center - 1), min(count, center + 1))
    background = np.concatenate(
        (
            np.arange(max(0, center - 10), max(0, center - 4)),
            np.arange(min(count, center + 5), min(count, center + 11)),
        )
    )
    if (
        target.size != TARGET_GATE_BINS
        or background.size < 3
        or np.intersect1d(target, background).size
    ):
        raise ValueError(
            f"Insufficient separated range bins around target bin {center}/{count}"
        )
    return target, background


def reduce_centered_doppler(
    power: np.ndarray,
    output_bins: int = DOPPLER_BINS,
) -> np.ndarray:
    power = np.asarray(power)
    if power.ndim != 1 or power.size % output_bins:
        raise ValueError(
            f"Cannot power-average Doppler shape {power.shape} to {output_bins} bins"
        )
    factor = power.size // output_bins
    return power.reshape(output_bins, factor).mean(axis=-1)


def doppler_cube_to_feature_step(
    doppler_cube: np.ndarray,
    target_range_bin: int,
) -> np.ndarray:
    """Build one notebook-equivalent [2, 64] live feature step."""
    doppler_cube = np.asarray(doppler_cube)
    if doppler_cube.ndim != 4 or not np.iscomplexobj(doppler_cube):
        raise ValueError(
            "Doppler cube must be complex with shape [doppler, tx, rx, range]"
        )
    if not np.isfinite(doppler_cube).all():
        raise ValueError("Doppler cube contains non-finite values")

    range_bins = int(doppler_cube.shape[-1])
    target_bins, background_bins = _range_indices(
        int(target_range_bin),
        range_bins,
    )
    target_power = (
        np.abs(doppler_cube[..., target_bins]) ** 2
    ).mean(axis=(1, 2, 3))
    background_power = (
        np.abs(doppler_cube[..., background_bins]) ** 2
    ).mean(axis=(1, 2, 3))
    target_power = reduce_centered_doppler(target_power)
    background_power = reduce_centered_doppler(background_power)
    log_target = np.log1p(target_power)
    step = np.stack(
        (log_target, log_target - np.log1p(background_power))
    ).astype(np.float32)
    if step.shape != (2, DOPPLER_BINS) or not np.isfinite(step).all():
        raise ValueError(f"Invalid live feature step: {step.shape}")
    return step


def _build_model(torch: Any, nn: Any, config: dict[str, Any]) -> Any:
    class DepthwiseSeparableResidual(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: tuple[int, int] = (1, 1),
        ) -> None:
            super().__init__()
            self.main = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            self.skip = (
                nn.Identity()
                if in_channels == out_channels and tuple(stride) == (1, 1)
                else nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                )
            )
            self.activation = nn.SiLU(inplace=True)

        def forward(self, inputs: Any) -> Any:
            return self.activation(self.main(inputs) + self.skip(inputs))

    class MicroDopplerCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            stem = int(config["stem_channels"])
            block_channels = [int(value) for value in config["block_channels"]]
            channels = [stem, *block_channels]
            self.stem = nn.Sequential(
                nn.Conv2d(
                    int(config["input_channels"]),
                    stem,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(stem),
                nn.SiLU(inplace=True),
            )
            self.blocks = nn.Sequential(
                *[
                    DepthwiseSeparableResidual(
                        channels[index],
                        channels[index + 1],
                        tuple(config["block_strides"][index]),
                    )
                    for index in range(len(block_channels))
                ]
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(float(config["dropout"]))
            self.classifier = nn.Linear(channels[-1], 1)

        def forward(self, inputs: Any) -> Any:
            if inputs.ndim != 4 or tuple(inputs.shape[1:]) != INPUT_SHAPE_CHW:
                raise ValueError(
                    f"Expected [batch, 2, 48, 64], received {tuple(inputs.shape)}"
                )
            features = self.blocks(self.stem(inputs))
            pooled = self.pool(features).flatten(1)
            return self.classifier(self.dropout(pooled)).squeeze(1)

    return MicroDopplerCNN()


class DroneBirdInference:
    """Stateful, CPU-only live inference for one consistently selected target."""

    def __init__(
        self,
        artifact_dir: Path,
        config: RadarInferenceConfig,
        profile_path: Path,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.config = config
        self.profile_path = Path(profile_path)
        self._history: deque[np.ndarray] = deque(maxlen=WINDOW_STEPS)

        try:
            import joblib
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "CNN classification requires joblib and scikit-learn"
            ) from exc
        try:
            import torch
            from torch import nn
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "CNN classification requires PyTorch 2.6 or newer; install "
                "'torch>=2.6,<3' or run with --no-classification"
            ) from exc
        try:
            torch_version = tuple(
                int(part)
                for part in torch.__version__.split("+", 1)[0].split(".")[:2]
            )
        except (AttributeError, ValueError):
            torch_version = (0, 0)
        if not (torch_version >= (2, 6) and torch_version < (3, 0)):
            raise RuntimeError(
                "CNN classification requires PyTorch >=2.6,<3; found "
                f"{getattr(torch, '__version__', 'unknown')}"
            )

        self._torch = torch
        state_path = self.artifact_dir / MODEL_STATE_NAME
        calibration_path = self.artifact_dir / CALIBRATION_NAME
        card_path = self.artifact_dir / MODEL_CARD_NAME
        for path in (state_path, calibration_path, card_path):
            if not path.is_file():
                raise FileNotFoundError(f"Classification artifact not found: {path}")

        checkpoint = torch.load(
            state_path,
            map_location="cpu",
            weights_only=True,
        )
        calibration = joblib.load(calibration_path)
        card = json.loads(card_path.read_text(encoding="utf-8"))
        self._validate_artifacts(checkpoint, calibration, card)

        self.threshold = float(calibration["threshold"])
        self.clip_low = self._channel_vector(calibration, "clip_low")
        self.clip_high = self._channel_vector(calibration, "clip_high")
        self.channel_mean = self._channel_vector(calibration, "channel_mean")
        self.channel_std = self._channel_vector(calibration, "channel_std")
        if np.any(self.clip_low > self.clip_high):
            raise ValueError("Classification clipping bounds are reversed")
        if np.any(self.channel_std <= 0.0):
            raise ValueError("Classification channel_std must be positive")
        self.calibrator = calibration["calibrator"]

        self.model = _build_model(torch, nn, checkpoint["cnn_config"])
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.to(torch.device("cpu"))
        self.model.eval()
        self.model_card = card

    @property
    def valid_steps(self) -> int:
        return len(self._history)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "model": "drone_bird_cnn",
            "backend": "pytorch",
            "device": "cpu",
            "feature_version": FEATURE_VERSION,
            "input_shape_chw": list(INPUT_SHAPE_CHW),
            "compatible_profile_sha256": normalized_profile_sha256(
                self.profile_path
            ),
            "threshold": self.threshold,
            "labels": ["drone", "not_drone", "unknown"],
            "negative_training_class": "others",
            "target_gate_range_bins": TARGET_GATE_BINS,
            "deployment_status": self.model_card.get(
                "deployment_status",
                DEPLOYMENT_STATUS,
            ),
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

    def update_feature_step(self, step: np.ndarray) -> InferenceResult:
        """Append one precomputed [2, 64] feature step and classify its window."""
        step = np.asarray(step, dtype=np.float32)
        if step.shape != (2, DOPPLER_BINS) or not np.isfinite(step).all():
            return self.reset(f"invalid_feature_step:{step.shape}")

        self._history.append(step)
        if len(self._history) < WINDOW_STEPS:
            return self.unknown("insufficient_history")

        try:
            window = np.stack(tuple(self._history), axis=1)
            probability = self.predict_feature_window(window)
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
            valid_steps=len(self._history),
        )

    def normalize_feature_window(self, window: np.ndarray) -> np.ndarray:
        """Normalize one raw [2, 48, 64] feature window for model inference."""
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
        normalized = normalized.astype(np.float32, copy=False)
        if not np.isfinite(normalized).all():
            raise ValueError("Normalized feature window contains non-finite values")
        return np.ascontiguousarray(normalized)

    def calibrate_logit(self, logit: float) -> float:
        """Apply the deployed probability calibrator to one model logit."""
        return float(
            self.calibrator.predict_proba(
                np.asarray([[float(logit)]], dtype=np.float64)
            )[0, 1]
        )

    def predict_feature_window(self, window: np.ndarray) -> float:
        """Classify one complete raw feature window without changing history."""
        normalized = self.normalize_feature_window(window)
        with self._torch.inference_mode():
            tensor = self._torch.from_numpy(normalized[None])
            logit = float(self.model(tensor).cpu().numpy()[0])
        return self.calibrate_logit(logit)

    def _validate_artifacts(
        self,
        checkpoint: Any,
        calibration: Any,
        card: Any,
    ) -> None:
        if not isinstance(checkpoint, dict):
            raise ValueError("CNN checkpoint must be a mapping")
        if checkpoint.get("architecture") != "MicroDopplerCNN":
            raise ValueError("Unsupported CNN architecture")
        if tuple(checkpoint.get("input_shape_chw", ())) != INPUT_SHAPE_CHW:
            raise ValueError("CNN checkpoint input shape is incompatible")
        if checkpoint.get("feature_version") != FEATURE_VERSION:
            raise ValueError("CNN checkpoint feature version is incompatible")
        if not isinstance(checkpoint.get("cnn_config"), dict):
            raise ValueError("CNN checkpoint is missing cnn_config")
        cnn_config = checkpoint["cnn_config"]
        expected_config = CNN_CONFIG
        observed_config = {
            "input_channels": cnn_config.get("input_channels"),
            "stem_channels": cnn_config.get("stem_channels"),
            "block_channels": list(cnn_config.get("block_channels", ())),
            "block_strides": [
                tuple(stride)
                for stride in cnn_config.get("block_strides", ())
            ],
            "dropout": cnn_config.get("dropout"),
        }
        if observed_config != expected_config:
            raise ValueError("CNN architecture configuration is incompatible")
        if int(checkpoint.get("target_gate_range_bins", -1)) != TARGET_GATE_BINS:
            raise ValueError("CNN target range-bin contract is incompatible")
        if not isinstance(checkpoint.get("state_dict"), dict):
            raise ValueError("CNN checkpoint is missing state_dict")
        if not isinstance(calibration, dict):
            raise ValueError("CNN calibration artifact must be a mapping")
        required = {
            "calibrator",
            "threshold",
            "clip_low",
            "clip_high",
            "channel_mean",
            "channel_std",
            "compatible_profile_sha256",
        }
        missing = required.difference(calibration)
        if missing:
            raise ValueError(
                "CNN calibration artifact is missing: "
                + ", ".join(sorted(missing))
            )
        threshold = float(calibration["threshold"])
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("CNN threshold must be a finite probability")
        observed_hash = normalized_profile_sha256(self.profile_path)
        if calibration["compatible_profile_sha256"] != observed_hash:
            raise ValueError(
                "Radar configuration mismatch: "
                f"model={calibration['compatible_profile_sha256']}, "
                f"observed={observed_hash}"
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
            raise ValueError(
                "Radar dimensions are incompatible with the CNN: "
                f"expected={expected_dimensions}, observed={observed_dimensions}"
            )
        if tuple(self.config.tx_channel_masks or ()) != (1, 4, 2):
            raise ValueError("Radar TX schedule is incompatible with the CNN")
        if not isinstance(card, dict):
            raise ValueError("CNN model card must be a mapping")
        if tuple(card.get("input_shape_chw", ())) != INPUT_SHAPE_CHW:
            raise ValueError("CNN model card input shape is incompatible")
        if card.get("feature_version") != FEATURE_VERSION:
            raise ValueError("CNN model card feature version is incompatible")
        if int(card.get("target_gate_range_bins", -1)) != TARGET_GATE_BINS:
            raise ValueError("CNN model card target range-bin contract is incompatible")
        if (
            card.get("compatible_profile_sha256")
            != calibration["compatible_profile_sha256"]
        ):
            raise ValueError(
                "CNN model card and calibration profile fingerprints differ"
            )

    @staticmethod
    def _channel_vector(
        calibration: dict[str, Any],
        key: str,
    ) -> np.ndarray:
        value = np.asarray(calibration[key], dtype=np.float32)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError(f"Classification {key} must contain two finite values")
        return value
