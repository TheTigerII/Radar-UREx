"""Two-transmitter phased-STT profile generation for the IWR6843ISK-ODS.

The standard IWR6843 supply configuration permits at most two simultaneous
transmitters.  This module therefore uses the horizontal TX0/TX1 pair only;
it never enables the three-transmitter mask.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np


PHASED_STT_OPERATION_MODE = "phased-array-stt"
PHASED_STT_TX_MASK = 0x3
PHASED_STT_MAX_AZIMUTH_DEG = 30.0
TX_PHASE_STEPS = 64
TX_PHASE_STEP_DEG = 360.0 / TX_PHASE_STEPS
TX0_TX1_SPACING_WAVELENGTHS = 1.0


@dataclass(frozen=True)
class PhasedSttProfile:
    path: Path
    requested_azimuth_deg: float
    requested_elevation_deg: float
    calibration_phase_deg: float
    steering_phase_deg: float
    programmed_phase_deg: float
    tx1_phase_code: int
    packed_phase_word: int


def _profile_coefficients(text: str) -> np.ndarray:
    commands = []
    for raw_line in text.splitlines():
        line = raw_line.split("%", 1)[0].split("#", 1)[0].strip()
        if line.startswith("compRangeBiasAndRxChanPhase "):
            commands.append(line.split())
    if len(commands) != 1 or len(commands[0]) != 26:
        raise ValueError(
            "Phased STT profile requires exactly one valid "
            "compRangeBiasAndRxChanPhase command"
        )
    values = np.asarray([float(value) for value in commands[0][2:]], dtype=float)
    coefficients = values[0::2] + 1j * values[1::2]
    if coefficients.shape != (12,) or not np.isfinite(coefficients).all():
        raise ValueError("Phased STT calibration coefficients are invalid")
    return coefficients


def tx_pair_calibration_phase_deg(coefficients: np.ndarray) -> float:
    """Estimate the TX1 phase correction from the calibrated virtual channels."""

    values = np.asarray(coefficients, dtype=np.complex128)
    if values.shape != (12,) or not np.isfinite(values).all():
        raise ValueError("TX phase calibration requires 12 finite coefficients")
    ratios = values[4:8] * np.conj(values[0:4])
    magnitudes = np.abs(ratios)
    if np.any(magnitudes <= np.finfo(float).tiny):
        raise ValueError("TX phase calibration contains a zero-valued channel")
    circular_mean = np.mean(ratios / magnitudes)
    if abs(circular_mean) <= np.finfo(float).eps:
        raise ValueError("TX phase calibration is not coherent across receivers")
    return float(np.angle(circular_mean, deg=True))


def tx1_steering_phase_deg(azimuth_deg: float, elevation_deg: float) -> float:
    """Return TX1 relative phase for the horizontal one-wavelength TX pair."""

    azimuth = float(azimuth_deg)
    elevation = float(elevation_deg)
    if not math.isfinite(azimuth) or not math.isfinite(elevation):
        raise ValueError("Phased STT beam angles must be finite")
    if abs(azimuth) > PHASED_STT_MAX_AZIMUTH_DEG:
        raise ValueError(
            "Phased STT azimuth must be between "
            f"-{PHASED_STT_MAX_AZIMUTH_DEG:g} and "
            f"+{PHASED_STT_MAX_AZIMUTH_DEG:g} degrees"
        )
    if abs(elevation) > 60.0:
        raise ValueError("Phased STT elevation must be between -60 and +60 degrees")
    direction_x = math.sin(math.radians(azimuth)) * math.cos(
        math.radians(elevation)
    )
    return -360.0 * TX0_TX1_SPACING_WAVELENGTHS * direction_x


def encode_tx1_phase_word(relative_phase_deg: float) -> tuple[int, int, float]:
    """Quantize TX1 phase and pack it into the SDK ``profileCfg`` word."""

    if not math.isfinite(relative_phase_deg):
        raise ValueError("TX phase must be finite")
    code = int(round(relative_phase_deg / TX_PHASE_STEP_DEG)) % TX_PHASE_STEPS
    # mmWaveLink's rlProfileCfg_t uses one byte per TX. TX0 remains zero and
    # TX1 occupies bits 15:8. Only the low six bits of that byte are used.
    packed_word = code << 8
    programmed = code * TX_PHASE_STEP_DEG
    return code, packed_word, programmed


def create_phased_stt_runtime_profile(
    source_path: Path | str,
    destination_path: Path | str,
    *,
    azimuth_deg: float,
    elevation_deg: float,
) -> PhasedSttProfile:
    """Write a temporary two-TX profile with the requested calibrated phase."""

    source = Path(source_path)
    destination = Path(destination_path)
    text = source.read_text(encoding="utf-8")
    coefficients = _profile_coefficients(text)
    calibration_phase = tx_pair_calibration_phase_deg(coefficients)
    steering_phase = tx1_steering_phase_deg(azimuth_deg, elevation_deg)
    code, packed_word, programmed_phase = encode_tx1_phase_word(
        calibration_phase + steering_phase
    )

    output: list[str] = []
    profile_count = 0
    channel_masks: list[int] = []
    chirp_masks: list[int] = []
    for raw_line in text.splitlines(keepends=True):
        body = raw_line.rstrip("\r\n")
        ending = raw_line[len(body) :]
        stripped = body.split("%", 1)[0].split("#", 1)[0].strip()
        tokens = stripped.split()
        if tokens and tokens[0] == "profileCfg":
            if len(tokens) < 15:
                raise ValueError("Malformed profileCfg in phased STT profile")
            tokens[7] = str(packed_word)
            indent = body[: len(body) - len(body.lstrip())]
            output.append(indent + " ".join(tokens) + ending)
            profile_count += 1
            continue
        if tokens and tokens[0] == "channelCfg":
            if len(tokens) < 3:
                raise ValueError("Malformed channelCfg in phased STT profile")
            channel_masks.append(int(tokens[2], 0))
        if tokens and tokens[0] == "chirpCfg":
            if len(tokens) < 9:
                raise ValueError("Malformed chirpCfg in phased STT profile")
            chirp_masks.append(int(tokens[8], 0))
        output.append(raw_line)

    if profile_count != 1:
        raise ValueError("Phased STT profile must contain exactly one profileCfg")
    if channel_masks != [PHASED_STT_TX_MASK]:
        raise ValueError(
            "Phased STT channelCfg must enable exactly TX0 and TX1 (mask 3)"
        )
    if chirp_masks != [PHASED_STT_TX_MASK]:
        raise ValueError(
            "Phased STT chirpCfg must use one simultaneous TX0+TX1 chirp (mask 3)"
        )

    marker = (
        "% hostPhasedStt "
        f"azimuthDeg {float(azimuth_deg):.7g} "
        f"elevationDeg {float(elevation_deg):.7g} "
        f"tx1CalibrationDeg {calibration_phase:.7g} "
        f"tx1SteeringDeg {steering_phase:.7g} "
        f"tx1Code {code} phaseWord {packed_word}\n"
    )
    output.insert(0, marker)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(output), encoding="utf-8")
    return PhasedSttProfile(
        path=destination,
        requested_azimuth_deg=float(azimuth_deg),
        requested_elevation_deg=float(elevation_deg),
        calibration_phase_deg=calibration_phase,
        steering_phase_deg=steering_phase,
        programmed_phase_deg=programmed_phase,
        tx1_phase_code=code,
        packed_phase_word=packed_word,
    )
