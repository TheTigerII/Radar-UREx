"""Opt-in detection enhancements for the conventional pipeline (GUI Mode 4).

Mode 1 forms its detection map by averaging power non-coherently over the 12
TDM-MIMO virtual channels, normalises it by a short exponentially averaged
clutter reference, and applies 2D OS-CFAR one frame at a time. This module adds
four independent options on top of that pipeline. Each defaults to off, so the
original Mode 1 path in ``dsp.py`` is never modified:

1. ``coherent-beam`` detection power: TDM-Doppler-compensated, calibrated
   12-channel receive beams are formed for every range-Doppler cell and the
   strongest beam inside the field of view becomes the detection statistic.
2. Track-before-detect: per-frame CFAR margins are integrated along physically
   reachable range paths so a persistent target that never crosses the
   single-frame threshold can still be declared.
3. Averaged clutter reference: the startup background is the arithmetic mean
   of the warm-up frames instead of a first-frame-seeded exponential average.
4. Calibrated CFAR threshold: the nominal OS-CFAR threshold is scaled to a
   Monte-Carlo-measured noise-only false-alarm probability of 1e-6 per cell.

The waveform, calibration, clustering, tracking and CNN contract are shared
with Mode 1, so range differences between Mode 1 and Mode 4 are attributable
to the options enabled here.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
from scipy import ndimage as scipy_ndimage

from main import dsp, dsp_kernels
from main.dsp import AdaptiveClutterMap, RadarDspConfig

DETECTION_POWER_NONCOHERENT = "noncoherent"
DETECTION_POWER_COHERENT_BEAM = "coherent-beam"
DETECTION_POWER_MODES = (DETECTION_POWER_NONCOHERENT, DETECTION_POWER_COHERENT_BEAM)

CLUTTER_REFERENCE_EMA = "ema"
CLUTTER_REFERENCE_MEAN = "mean"
CLUTTER_REFERENCE_MODES = (CLUTTER_REFERENCE_EMA, CLUTTER_REFERENCE_MEAN)

# Eight-point spatial FFT on the 4 x 4 ODS virtual grid. Beam spacing is 0.25
# in direction cosine against a ~0.44 half-power beamwidth, which bounds the
# straddle loss to roughly 1 dB while keeping the beam bank small (37 beams
# inside +/-60 degrees and the visible region).
COHERENT_BEAM_FFT_SIZE = 8

# Mode 1 runs OS-CFAR with a nominal Pfa of 1e-3, but the multiplier assumes a
# single-look exponential cell while the statistic is a 12-channel average (or
# a beam maximum), and a cell must also pass both the range and Doppler tests,
# be a Doppler peak and exceed the clutter-map SNR floor. In thermal noise the
# resulting empirical Pfa is far below 1e-7, so the threshold is several dB
# higher than needed. "calibrated" scales the nominal threshold by a factor
# measured by Monte Carlo on noise-only frames for an empirical per-cell Pfa of
# 1e-6 (~0.007 false cells per 128 x 64 frame, removed by track confirmation).
# Keys are (detection_power, clutter_reference). Re-measure with
# ``scripts/calibrate_enhanced_cfar.py`` if the waveform or CFAR windows change.
CFAR_THRESHOLD_NOMINAL = "nominal"
CFAR_THRESHOLD_CALIBRATED = "calibrated"
CFAR_THRESHOLD_MODES = (CFAR_THRESHOLD_NOMINAL, CFAR_THRESHOLD_CALIBRATED)
NOMINAL_FALSE_ALARM_RATE = 1e-3
CALIBRATED_PFA = 1e-6
# Measured on 8e6 noise-only cells per configuration (128 Doppler x 64 range,
# 30-frame clutter warm-up, OS-CFAR windows as in Mode 1). In the same
# simulation the combined Mode 4 configuration needed 10.3-11.6 dB less target
# power than Mode 1 for Pd = 0.9-0.5 (thermal noise only; real clutter is
# harsher, so treat this as an upper bound to be checked in the field).
CALIBRATED_THRESHOLD_MULTIPLIER = {
    (DETECTION_POWER_NONCOHERENT, CLUTTER_REFERENCE_EMA): 0.199,
    (DETECTION_POWER_NONCOHERENT, CLUTTER_REFERENCE_MEAN): 0.165,
    (DETECTION_POWER_COHERENT_BEAM, CLUTTER_REFERENCE_EMA): 0.317,
    (DETECTION_POWER_COHERENT_BEAM, CLUTTER_REFERENCE_MEAN): 0.259,
}

# Track-before-detect defaults, in units of the (calibrated) detection
# threshold. With the full Mode 4 configuration, 1,500 simulated noise-only
# frames produced no declarations at 0.50 (0.45 gave ~1 per second), while a
# target detected by single-frame CFAR in 3% / 17% of frames was detected in
# 78% / 96% (about 3.6 dB at Pd = 0.5). Eight frames is ~0.27 s at 30 Hz.
DEFAULT_TBD_WINDOW_FRAMES = 8
DEFAULT_TBD_MAX_JUMP_BINS = 2
DEFAULT_TBD_INTEGRATED_THRESHOLD = 0.50
DEFAULT_TBD_CURRENT_FRAME_MINIMUM = 0.25
DEFAULT_TBD_SCORE_CLIP = 1.5
TBD_SUPPRESSION_RADIUS_BINS = 2


@dataclass(frozen=True)
class EnhancedDetectionOptions:
    detection_power: str = DETECTION_POWER_NONCOHERENT
    tbd_window_frames: int = 0
    clutter_reference: str = CLUTTER_REFERENCE_EMA
    cfar_threshold: str = CFAR_THRESHOLD_NOMINAL

    def __post_init__(self) -> None:
        if self.detection_power not in DETECTION_POWER_MODES:
            raise ValueError(f"Unknown detection power mode: {self.detection_power}")
        if self.clutter_reference not in CLUTTER_REFERENCE_MODES:
            raise ValueError(f"Unknown clutter reference: {self.clutter_reference}")
        if self.cfar_threshold not in CFAR_THRESHOLD_MODES:
            raise ValueError(f"Unknown CFAR threshold mode: {self.cfar_threshold}")
        if self.tbd_window_frames < 0 or self.tbd_window_frames == 1:
            raise ValueError("Track-before-detect window must be 0 (off) or >= 2")

    @property
    def enabled(self) -> bool:
        return self != EnhancedDetectionOptions()

    @property
    def threshold_multiplier(self) -> float:
        if self.cfar_threshold == CFAR_THRESHOLD_NOMINAL:
            return 1.0
        value = CALIBRATED_THRESHOLD_MULTIPLIER[
            (self.detection_power, self.clutter_reference)
        ]
        if value is None:
            raise ValueError("Calibrated CFAR multiplier has not been measured")
        return float(value)

    def describe(self) -> str:
        return (
            f"detection_power={self.detection_power}, "
            f"clutter_reference={self.clutter_reference}, "
            f"cfar_threshold={self.cfar_threshold} "
            f"(x{self.threshold_multiplier:.3f}), "
            f"tbd_window_frames={self.tbd_window_frames}"
        )


MODE4_OPTIONS = EnhancedDetectionOptions(
    detection_power=DETECTION_POWER_COHERENT_BEAM,
    tbd_window_frames=DEFAULT_TBD_WINDOW_FRAMES,
    clutter_reference=CLUTTER_REFERENCE_MEAN,
    cfar_threshold=CFAR_THRESHOLD_CALIBRATED,
)


class MeanClutterMap(AdaptiveClutterMap):
    """Clutter map whose warm-up reference is the arithmetic mean of frames.

    ``AdaptiveClutterMap`` seeds its reference with the first frame and then
    applies a 0.02 exponential update. After the default 30-frame warm-up the
    first frame still carries ~56% of the weight (about three effective
    frames), so every later frame is divided by a noisy reference. Averaging
    all warm-up frames equally gives the reference ``warmup_frames`` effective
    looks. Behaviour after warm-up is inherited unchanged.
    """

    def update(
        self,
        power_map: np.ndarray,
        protected_detections: Optional[np.ndarray] = None,
    ) -> None:
        if self.is_ready:
            super().update(power_map, protected_detections)
            return
        power = self._validated_power(power_map)
        self._ensure_shape(power.shape)
        assert self._background_power is not None
        count = self._frames_seen
        self._background_power *= count / (count + 1.0)
        self._background_power += power / (count + 1.0)
        self._frames_seen = count + 1


def create_clutter_map(
    reference: str,
    *,
    update_rate: float,
    warmup_frames: int,
    minimum_snr_db: float,
) -> AdaptiveClutterMap:
    cls = MeanClutterMap if reference == CLUTTER_REFERENCE_MEAN else AdaptiveClutterMap
    return cls(
        update_rate=update_rate,
        warmup_frames=warmup_frames,
        minimum_snr_db=minimum_snr_db,
        adapt_after_warmup=False,
    )


def tdm_doppler_compensation(doppler_cube: np.ndarray) -> np.ndarray:
    """Remove the Doppler-induced phase between TDM transmit slots.

    ``doppler_cube`` is the fftshifted ``[doppler, tx slot, rx, range]`` output
    of ``dsp_kernels.doppler_fft``. A target in shifted Doppler bin ``f`` is
    sampled ``k`` chirp periods later in TX slot ``k`` and therefore carries an
    extra ``exp(j*2*pi*f*k/(D*T))`` phase. Without this correction the virtual
    array is not coherent for moving targets.
    """
    if doppler_cube.ndim != 4:
        raise ValueError("Doppler cube must have shape [doppler, tx, rx, range]")
    doppler_bins, tx_slots = doppler_cube.shape[:2]
    correction = _tdm_correction(doppler_bins, tx_slots)
    return doppler_cube * correction[:, :, np.newaxis, np.newaxis]


@lru_cache(maxsize=16)
def _tdm_correction(doppler_bins: int, tx_slots: int) -> np.ndarray:
    frequencies = np.arange(doppler_bins, dtype=np.float64) - doppler_bins // 2
    slots = np.arange(tx_slots, dtype=np.float64)
    phase = -2.0 * np.pi * np.outer(frequencies, slots) / (doppler_bins * tx_slots)
    correction = np.exp(1j * phase).astype(np.complex64)
    correction.setflags(write=False)
    return correction


@lru_cache(maxsize=8)
def coherent_beam_steering(
    fft_size: int = COHERENT_BEAM_FFT_SIZE,
    fov_deg: float = 60.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(weights[16, beams], azimuth_u, elevation_u)`` for the 4x4 grid.

    The weights reproduce the fftshifted 2D spatial FFT used by
    ``dsp.estimate_xyz_from_virtual_arrays`` so beam directions share its
    direction-cosine convention.
    """
    shifted = np.arange(fft_size) - fft_size // 2
    direction = 2.0 * shifted / float(fft_size)
    limit = float(np.sin(np.deg2rad(min(max(fov_deg, 0.0), 90.0)))) + 1e-9
    beams = [
        (kr, kc)
        for kr, v in zip(shifted, direction)
        for kc, u in zip(shifted, direction)
        if abs(u) <= limit and abs(v) <= limit and u * u + v * v <= 1.0 + 1e-9
    ]
    rows, cols = np.divmod(np.arange(16), 4)
    kr = np.array([b[0] for b in beams], dtype=np.float64)
    kc = np.array([b[1] for b in beams], dtype=np.float64)
    phase = -2.0 * np.pi * (
        np.outer(rows, kr) + np.outer(cols, kc)
    ) / float(fft_size)
    weights = np.exp(1j * phase).astype(np.complex64)
    weights.setflags(write=False)
    return weights, 2.0 * kc / fft_size, 2.0 * kr / fft_size


def coherent_beam_power(
    compensated_cube: np.ndarray,
    config: RadarDspConfig,
    *,
    fov_deg: float = 60.0,
    fft_size: int = COHERENT_BEAM_FFT_SIZE,
) -> np.ndarray:
    """Return ``[doppler, range]`` maximum calibrated beam power.

    Power is divided by the number of virtual channels so uncorrelated noise
    has the same mean as Mode 1's non-coherent channel average, while a
    coherent target gains up to 10*log10(12) = 10.8 dB.
    """
    doppler_bins, tx_slots, rx_count, range_bins = compensated_cube.shape
    channels = tx_slots * rx_count
    # Push one unit sample per virtual channel through the Mode 1 grid builder
    # so the array layout and calibration coefficients are taken from the same
    # code the angle estimator uses, then fold them into the beam weights.
    basis = np.eye(channels, dtype=np.complex64).reshape(channels, tx_slots, rx_count)
    grids = dsp.build_virtual_antenna_grids(basis, config)
    if grids.shape[1:] != (4, 4):
        # Non-ODS layouts fall back to the non-coherent statistic.
        return np.mean(np.abs(compensated_cube) ** 2, axis=(1, 2))
    steering, _, _ = coherent_beam_steering(fft_size, float(fov_deg))
    weights = np.ascontiguousarray(
        (grids.reshape(channels, 16) @ steering).astype(np.complex64)
    )
    samples = np.ascontiguousarray(
        compensated_cube.transpose(0, 3, 1, 2), dtype=np.complex64
    ).reshape(doppler_bins * range_bins, channels)
    responses = samples @ weights
    power = (responses.real**2 + responses.imag**2).max(axis=1) / float(channels)
    return power.reshape(doppler_bins, range_bins)


def os_cfar_thresholds(
    power_map: np.ndarray,
    *,
    false_alarm_rate: float,
    range_guard_cells: int,
    doppler_guard_cells: int,
    range_training_cells: int,
    doppler_training_cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(threshold, valid)`` reproducing ``dsp_kernels.os_cfar_2d``.

    ``power > threshold`` restricted to ``valid`` equals the boolean output of
    ``os_cfar_2d`` with the same arguments; the threshold itself is needed to
    express per-cell CFAR margins for track-before-detect.
    """
    power = np.maximum(np.asarray(power_map, dtype=np.float64), 0.0)
    valid = np.zeros(power.shape, dtype=bool)
    if power.ndim != 2 or power.size == 0:
        return np.full(power.shape, np.inf), valid
    doppler_bins, range_bins = power.shape
    range_margin = range_guard_cells + range_training_cells
    doppler_margin = doppler_guard_cells + doppler_training_cells
    if range_bins <= 2 * range_margin or doppler_bins <= 2 * doppler_margin:
        return np.full(power.shape, np.inf), valid
    pfa = min(max(false_alarm_rate, 1e-9), 0.5)

    def parameters(training: int) -> tuple[int, float]:
        total = 2 * training
        rank = max(1, int(np.ceil(0.75 * total)))
        return rank - 1, dsp_kernels._os_scale(total, rank, pfa)

    range_rank, range_scale = parameters(range_training_cells)
    doppler_rank, doppler_scale = parameters(doppler_training_cells)
    range_thresholds = dsp_kernels._os_thresholds_along_axis(
        power,
        axis=1,
        guard_cells=range_guard_cells,
        training_cells=range_training_cells,
        rank_index=range_rank,
        scale=range_scale,
    )
    doppler_thresholds = dsp_kernels._os_thresholds_along_axis(
        power,
        axis=0,
        guard_cells=doppler_guard_cells,
        training_cells=doppler_training_cells,
        rank_index=doppler_rank,
        scale=doppler_scale,
    )
    valid[:, range_margin : range_bins - range_margin] = True
    return np.maximum(range_thresholds, doppler_thresholds), valid


@dataclass(frozen=True)
class TrackBeforeDetectResult:
    detected: bool
    doppler_bin: Optional[int]
    range_bin: Optional[int]
    integrated_score: Optional[float]
    window_frames: int


class TrackBeforeDetect:
    """Integrate CFAR margins along reachable range-Doppler paths.

    Each frame contributes a ``[doppler, range]`` map of ``power / detection
    threshold`` (clipped, so one strong clutter flash cannot carry a window).
    A max-sum dynamic programme over the last ``window_frames`` maps lets the
    path move at most ``max_jump_doppler_bins`` / ``max_jump_range_bins`` per
    frame; Doppler wraps. Restricting both axes matters: integrating only the
    per-range maximum over all Doppler bins lifts the noise-only path score to
    ~0.55 of the threshold and leaves little room for a weak target.
    A detection needs the path mean and the current frame's margin to clear
    their thresholds.
    """

    def __init__(
        self,
        window_frames: int = DEFAULT_TBD_WINDOW_FRAMES,
        *,
        max_jump_range_bins: int = DEFAULT_TBD_MAX_JUMP_BINS,
        max_jump_doppler_bins: int = DEFAULT_TBD_MAX_JUMP_BINS,
        integrated_threshold: float = DEFAULT_TBD_INTEGRATED_THRESHOLD,
        current_frame_minimum: float = DEFAULT_TBD_CURRENT_FRAME_MINIMUM,
        score_clip: float = DEFAULT_TBD_SCORE_CLIP,
    ) -> None:
        if window_frames < 2:
            raise ValueError("Track-before-detect needs a window of at least 2")
        self.window_frames = int(window_frames)
        self.max_jump_range_bins = max(int(max_jump_range_bins), 0)
        self.max_jump_doppler_bins = max(int(max_jump_doppler_bins), 0)
        self.integrated_threshold = float(integrated_threshold)
        self.current_frame_minimum = float(current_frame_minimum)
        self.score_clip = float(score_clip)
        self._history: deque[np.ndarray] = deque(maxlen=self.window_frames)

    def reset(self) -> None:
        self._history.clear()

    def update(self, margin_map: np.ndarray) -> TrackBeforeDetectResult:
        margins = np.clip(
            np.nan_to_num(np.asarray(margin_map, dtype=np.float32)),
            0.0,
            self.score_clip,
        )
        if margins.ndim != 2:
            raise ValueError("Margin map must have shape [doppler, range]")
        if self._history and self._history[-1].shape != margins.shape:
            self._history.clear()
        self._history.append(margins)
        frames = len(self._history)
        if frames < self.window_frames:
            return TrackBeforeDetectResult(False, None, None, None, frames)

        size = (2 * self.max_jump_doppler_bins + 1, 2 * self.max_jump_range_bins + 1)
        accumulated = self._history[0].copy()
        for scores in list(self._history)[1:]:
            reachable = scipy_ndimage.maximum_filter(
                accumulated,
                size=size,
                mode=("wrap", "constant"),
                cval=0.0,
            )
            accumulated = scores + reachable
        mean_score = accumulated / frames
        candidate = np.where(margins >= self.current_frame_minimum, mean_score, -np.inf)
        flat = int(np.argmax(candidate))
        doppler_bin, range_bin = np.unravel_index(flat, candidate.shape)
        score = float(candidate[doppler_bin, range_bin])
        if not np.isfinite(score):
            return TrackBeforeDetectResult(False, None, None, None, frames)
        return TrackBeforeDetectResult(
            score >= self.integrated_threshold,
            int(doppler_bin),
            int(range_bin),
            score,
            frames,
        )


@dataclass(frozen=True)
class EnhancedDetectionDiagnostics:
    cfar_candidates: int
    tbd_injected: bool
    tbd_integrated_score: Optional[float]
    tbd_range_bin: Optional[int]


def compute_enhanced_point_cloud(
    range_fft: np.ndarray,
    range_axis: Optional[np.ndarray],
    config: RadarDspConfig,
    options: EnhancedDetectionOptions,
    *,
    doppler_cube: Optional[np.ndarray] = None,
    clutter_map: Optional[AdaptiveClutterMap] = None,
    track_before_detect: Optional[TrackBeforeDetect] = None,
    range_guard_cells: int = 2,
    doppler_guard_cells: int = 1,
    range_training_cells: int = 4,
    doppler_training_cells: int = 2,
    min_range_m: float = 0.25,
    max_range_m: float = 10.0,
    azimuth_fov_deg: float = 60.0,
    elevation_fov_deg: float = 60.0,
) -> tuple[np.ndarray, EnhancedDetectionDiagnostics]:
    """Mode 4 counterpart of ``dsp.compute_point_cloud``.

    Returns the same ``[N, 4]`` (x, y, z, dB) point array so clustering,
    tracking and classification are unchanged.
    """
    empty = np.empty((0, 4), dtype=np.float32)
    if doppler_cube is None:
        doppler_cube = dsp.compute_range_doppler_fft(range_fft, config)
    coherent = options.detection_power == DETECTION_POWER_COHERENT_BEAM
    if coherent:
        doppler_cube = tdm_doppler_compensation(doppler_cube)
        raw_detection_power = coherent_beam_power(
            doppler_cube,
            config,
            fov_deg=max(azimuth_fov_deg, elevation_fov_deg),
        )
    else:
        raw_detection_power = np.mean(np.abs(doppler_cube) ** 2, axis=(1, 2))
    if raw_detection_power.size == 0:
        return empty, EnhancedDetectionDiagnostics(0, False, None, None)

    detection_power = (
        clutter_map.normalize(raw_detection_power)
        if clutter_map is not None
        else raw_detection_power
    )
    threshold, valid = os_cfar_thresholds(
        detection_power,
        false_alarm_rate=NOMINAL_FALSE_ALARM_RATE,
        range_guard_cells=range_guard_cells,
        doppler_guard_cells=doppler_guard_cells,
        range_training_cells=range_training_cells,
        doppler_training_cells=doppler_training_cells,
    )
    threshold = threshold * options.threshold_multiplier
    detections = valid & (detection_power > threshold)
    if clutter_map is not None:
        detections &= detection_power >= clutter_map.minimum_snr_linear
    peaks = dsp.doppler_peak_mask(detection_power)
    detections &= peaks
    detections = dsp._apply_min_range_gate(detections, range_axis, min_range_m)
    if clutter_map is not None:
        clutter_map.update(raw_detection_power, detections)

    if range_axis is not None and range_axis.size == detection_power.shape[1]:
        range_values = np.asarray(range_axis, dtype=np.float64)
    else:
        range_values = np.arange(detection_power.shape[1], dtype=np.float64)
    in_gate = valid.any(axis=0)
    if min_range_m > 0.0:
        in_gate &= range_values >= min_range_m
    if max_range_m > 0.0:
        in_gate &= range_values <= max_range_m

    candidate_indices = np.argwhere(detections)
    tbd_injected = False
    tbd_score: Optional[float] = None
    tbd_bin: Optional[int] = None
    ready = clutter_map is None or clutter_map.is_ready
    if track_before_detect is not None and ready:
        margin = np.where(
            valid & peaks & in_gate[np.newaxis, :],
            detection_power / np.maximum(threshold, np.finfo(float).tiny),
            0.0,
        )
        result = track_before_detect.update(margin)
        tbd_score = result.integrated_score
        tbd_bin = result.range_bin
        if (
            result.detected
            and result.range_bin is not None
            and result.doppler_bin is not None
        ):
            near_existing = candidate_indices.size and np.any(
                np.abs(candidate_indices[:, 1] - result.range_bin)
                <= TBD_SUPPRESSION_RADIUS_BINS
            )
            if not near_existing:
                candidate_indices = np.vstack(
                    (
                        candidate_indices.reshape(-1, 2),
                        np.array([[result.doppler_bin, result.range_bin]]),
                    )
                )
                tbd_injected = True

    cfar_count = int(np.count_nonzero(detections))
    diagnostics = EnhancedDetectionDiagnostics(cfar_count, tbd_injected, tbd_score, tbd_bin)
    if candidate_indices.size == 0:
        return empty, diagnostics

    doppler_bins = candidate_indices[:, 0]
    range_bins = candidate_indices[:, 1]
    candidate_powers = raw_detection_power[doppler_bins, range_bins]
    target_ranges_m = range_values[range_bins]
    if max_range_m > 0.0:
        keep = target_ranges_m <= max_range_m
        doppler_bins = doppler_bins[keep]
        range_bins = range_bins[keep]
        candidate_powers = candidate_powers[keep]
        target_ranges_m = target_ranges_m[keep]
    if target_ranges_m.size == 0:
        return empty, diagnostics

    virtual_samples = doppler_cube[doppler_bins, :, :, range_bins]
    xyz_m, angle_valid = dsp.estimate_xyz_from_virtual_arrays(
        virtual_samples,
        target_ranges_m,
        config,
    )
    keep = angle_valid & dsp._points_are_within_fov(
        xyz_m,
        azimuth_fov_deg=azimuth_fov_deg,
        elevation_fov_deg=elevation_fov_deg,
    )
    xyz_m = xyz_m[keep]
    candidate_powers = candidate_powers[keep]
    if xyz_m.size == 0:
        return empty, diagnostics
    magnitude_db = 10.0 * np.log10(candidate_powers + 1e-12)
    points = np.column_stack((xyz_m, magnitude_db)).astype(np.float32, copy=False)
    return points, diagnostics
