"""
Feature Extraction Module for the Exoplanet Detection Pipeline.

Creates input features for the ML classifier from detected transit signals:
  - Global view: binned phase-folded light curve spanning the full orbit.
  - Local view: zoomed-in phase-folded view centred on the transit.
  - Numerical features: transit depth, duration, period, SDE, SNR, V-shape
    metric, odd/even depth ratio, secondary eclipse depth, and
    duration/period ratio.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.utils import logger, timer, phase_fold, bin_data



def create_global_view(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    n_bins: int = None,
) -> np.ndarray:
    """Create a binned phase-folded light curve spanning the full orbit.

    Parameters
    ----------
    time : np.ndarray
        Time stamps (BJD or equivalent).
    flux : np.ndarray
        Corresponding flux values (already normalised to ~1).
    period : float
        Orbital period in days.
    epoch : float
        Reference mid-transit time (T0).
    n_bins : int, optional
        Number of equally-spaced phase bins.  Defaults to
        ``config.GLOBAL_VIEW_BINS`` (201).

    Returns
    -------
    np.ndarray
        1-D array of shape ``(n_bins,)`` with the binned flux values,
        normalised to a zero baseline (median subtracted).
    """
    if n_bins is None:
        n_bins = config.GLOBAL_VIEW_BINS

    # Validate inputs
    if len(time) == 0 or len(flux) == 0:
        logger.warning("create_global_view: empty time/flux arrays")
        return np.zeros(n_bins)

    if period <= 0:
        logger.warning("create_global_view: non-positive period (%.6f)", period)
        return np.zeros(n_bins)

    # Remove NaN/inf values from the input
    valid = np.isfinite(time) & np.isfinite(flux)
    if valid.sum() < 3:
        logger.warning("create_global_view: fewer than 3 valid data points")
        return np.zeros(n_bins)

    time_clean = time[valid]
    flux_clean = flux[valid]

    # Phase-fold
    phase = phase_fold(time_clean, period, epoch)

    # Bin into equally-spaced phase bins over [-0.5, 0.5]
    _, bin_means, _ = bin_data(phase, flux_clean, n_bins, x_range=(-0.5, 0.5))

    # Interpolate NaN bins (empty bins with no data points)
    bin_means = _interpolate_nans(bin_means)

    # Normalise to zero baseline (subtract median so out-of-transit ≈ 0)
    baseline = np.nanmedian(bin_means)
    bin_means -= baseline

    return bin_means


def create_local_view(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    duration: float,
    n_bins: int = None,
    half_width: float = None,
) -> np.ndarray:
    """Create a zoomed-in phase-folded view centred on the transit.

    Only data points within ±``half_width * duration`` of the transit
    centre (phase = 0) are retained and binned, capturing the detailed
    transit shape for CNN input.

    Parameters
    ----------
    time : np.ndarray
        Time stamps.
    flux : np.ndarray
        Corresponding flux values.
    period : float
        Orbital period in days.
    epoch : float
        Reference mid-transit time.
    duration : float
        Transit duration in days.
    n_bins : int, optional
        Number of phase bins.  Defaults to ``config.LOCAL_VIEW_BINS`` (61).
    half_width : float, optional
        View half-width in units of *duration*.  Defaults to
        ``config.LOCAL_VIEW_HALF_WIDTH`` (2.0).

    Returns
    -------
    np.ndarray
        1-D array of shape ``(n_bins,)`` with the binned flux values for
        the local transit view, normalised to zero baseline.
    """
    if n_bins is None:
        n_bins = config.LOCAL_VIEW_BINS
    if half_width is None:
        half_width = config.LOCAL_VIEW_HALF_WIDTH

    # Validate inputs
    if len(time) == 0 or len(flux) == 0:
        logger.warning("create_local_view: empty time/flux arrays")
        return np.zeros(n_bins)

    if period <= 0 or duration <= 0:
        logger.warning(
            "create_local_view: non-positive period (%.6f) or duration (%.6f)",
            period, duration,
        )
        return np.zeros(n_bins)

    # Remove NaN/inf
    valid = np.isfinite(time) & np.isfinite(flux)
    if valid.sum() < 3:
        logger.warning("create_local_view: fewer than 3 valid data points")
        return np.zeros(n_bins)

    time_clean = time[valid]
    flux_clean = flux[valid]

    # Phase-fold
    phase = phase_fold(time_clean, period, epoch)

    # Convert half-width from duration units to phase units
    phase_half_width = (half_width * duration) / period

    # Select points within the local window around transit centre
    local_mask = np.abs(phase) <= phase_half_width
    if local_mask.sum() < 3:
        logger.warning("create_local_view: fewer than 3 points in local window")
        return np.zeros(n_bins)

    local_phase = phase[local_mask]
    local_flux = flux_clean[local_mask]

    # Bin
    _, bin_means, _ = bin_data(
        local_phase, local_flux, n_bins,
        x_range=(-phase_half_width, phase_half_width),
    )

    # Interpolate NaN bins
    bin_means = _interpolate_nans(bin_means)

    # Normalise to zero baseline
    baseline = np.nanmedian(bin_means)
    bin_means -= baseline

    return bin_means



def extract_features(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    detection_result: Dict[str, Any],
) -> Dict[str, float]:
    """Extract numerical features from a detected transit signal.

    Parameters
    ----------
    time : np.ndarray
        Time stamps.
    flux : np.ndarray
        Corresponding flux values.
    flux_err : np.ndarray
        Flux uncertainties (1-σ).
    detection_result : dict
        Dictionary produced by the signal-detection stage.  Expected keys:
        ``period``, ``epoch`` (or ``t0``), ``duration``, ``depth``, ``sde``.

    Returns
    -------
    dict
        Dictionary with the following feature keys:

        * ``transit_depth`` – depth of the primary dip
        * ``transit_duration`` – duration in days
        * ``period`` – orbital period in days
        * ``sde`` – Signal Detection Efficiency
        * ``snr`` – transit signal-to-noise ratio
        * ``v_shape`` – V-shapedness metric (0 = flat-bottomed, 1 = pure V)
        * ``odd_even_ratio`` – ratio of odd to even transit depths
        * ``secondary_depth`` – depth of secondary eclipse (phase ≈ 0.5)
        * ``duration_period_ratio`` – duration / period
    """
    # Unpack detection result (tolerate 't0' or 'epoch')
    period = float(detection_result.get("period", 0.0))
    epoch = float(detection_result.get("epoch", detection_result.get("t0", 0.0)))
    duration = float(detection_result.get("duration", 0.0))
    depth = float(detection_result.get("depth", 0.0))
    sde = float(detection_result.get("sde", 0.0))

    # --- Remove invalid data points ---
    valid = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    time_v = time[valid]
    flux_v = flux[valid]
    flux_err_v = flux_err[valid]

    # Phase-fold for downstream helpers
    phase = phase_fold(time_v, period, epoch) if period > 0 else np.zeros_like(time_v)

    # --- SNR ---
    snr = _compute_snr(time_v, flux_v, flux_err_v, period, epoch, duration, depth)

    # --- V-shape ---
    v_shape = compute_v_shape(phase, flux_v, duration / period if period > 0 else 0.0)

    # --- Odd / even depth ratio ---
    odd_even_ratio = compute_odd_even_depth(time_v, flux_v, period, epoch, duration)

    # --- Secondary eclipse depth ---
    secondary_depth = compute_secondary_eclipse_depth(phase, flux_v)

    # --- Duration / period ratio ---
    duration_period_ratio = duration / period if period > 0 else 0.0

    features: Dict[str, float] = {
        "transit_depth": depth,
        "transit_duration": duration,
        "period": period,
        "sde": sde,
        "snr": snr,
        "v_shape": v_shape,
        "odd_even_ratio": odd_even_ratio,
        "secondary_depth": secondary_depth,
        "duration_period_ratio": duration_period_ratio,
    }

    return features



def compute_v_shape(
    phase: np.ndarray,
    flux: np.ndarray,
    duration: float,
) -> float:
    """Compute the V-shape metric from a phase-folded transit.

    The V-shapedness is defined as::

        V = 1 - (flat_bottom_duration / total_transit_duration)

    A perfectly flat-bottomed transit gives V ≈ 0, while a pure V-shaped
    eclipse gives V ≈ 1.  Higher values are more indicative of eclipsing
    binaries.

    Parameters
    ----------
    phase : np.ndarray
        Phase values in [-0.5, 0.5] (transit centred at 0).
    flux : np.ndarray
        Flux values corresponding to ``phase``.
    duration : float
        Transit duration **in phase units** (i.e. duration_days / period).

    Returns
    -------
    float
        V-shape metric in [0, 1].  Returns 0.5 when the measurement is
        not possible (insufficient data).
    """
    if duration <= 0 or len(phase) < 5:
        return 0.5  # default / indeterminate

    # Select in-transit points
    half_dur = duration / 2.0
    in_transit = np.abs(phase) <= half_dur
    if in_transit.sum() < 5:
        return 0.5

    transit_flux = flux[in_transit]
    transit_phase = phase[in_transit]

    # Determine the depth at the very bottom of the transit
    depth_full = np.nanmedian(flux[~in_transit]) - np.nanmin(transit_flux)
    if depth_full <= 0:
        return 0.5

    # Threshold for "flat bottom": flux within 10 % of the minimum
    bottom_threshold = np.nanmin(transit_flux) + 0.1 * depth_full
    flat_mask = transit_flux <= bottom_threshold

    if flat_mask.sum() < 2:
        # No measurable flat bottom → very V-shaped
        return 1.0

    # Flat-bottom duration as a fraction of total transit duration
    flat_phase_range = np.ptp(transit_phase[flat_mask])
    total_phase_range = np.ptp(transit_phase)

    if total_phase_range <= 0:
        return 0.5

    v_shape = 1.0 - (flat_phase_range / total_phase_range)
    return float(np.clip(v_shape, 0.0, 1.0))


def compute_odd_even_depth(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    duration: float,
) -> float:
    """Compare transit depths of odd- and even-numbered transits.

    For a genuine planet the odd and even depths should be equal.  A
    significant difference indicates an eclipsing binary whose primary
    and secondary eclipses have different depths.

    Parameters
    ----------
    time : np.ndarray
        Time stamps.
    flux : np.ndarray
        Flux values.
    period : float
        Orbital period in days.
    epoch : float
        Reference mid-transit time.
    duration : float
        Transit duration in days.

    Returns
    -------
    float
        Ratio of the deeper transit depth to the shallower one.  Returns
        1.0 when the measurement is not possible.  Values far from 1
        suggest an eclipsing binary.
    """
    if period <= 0 or duration <= 0 or len(time) < 10:
        return 1.0

    # Assign transit numbers
    transit_number = np.round((time - epoch) / period).astype(int)
    phase = phase_fold(time, period, epoch)

    # In-transit mask (within ±half duration in phase)
    phase_half_dur = (duration / 2.0) / period
    in_transit = np.abs(phase) <= phase_half_dur

    if in_transit.sum() < 4:
        return 1.0

    # Out-of-transit baseline
    oot_mask = ~in_transit
    if oot_mask.sum() < 5:
        return 1.0
    baseline = np.nanmedian(flux[oot_mask])

    # Split into odd/even
    odd_mask = in_transit & ((transit_number % 2) != 0)
    even_mask = in_transit & ((transit_number % 2) == 0)

    if odd_mask.sum() < 2 or even_mask.sum() < 2:
        return 1.0

    odd_depth = baseline - np.nanmedian(flux[odd_mask])
    even_depth = baseline - np.nanmedian(flux[even_mask])

    # Avoid division by zero
    if even_depth <= 0 and odd_depth <= 0:
        return 1.0

    # Ratio: deeper / shallower (always ≥ 1)
    depths = sorted([abs(odd_depth), abs(even_depth)])
    if depths[0] <= 0:
        return 1.0

    return float(depths[1] / depths[0])


def compute_secondary_eclipse_depth(
    phase: np.ndarray,
    flux: np.ndarray,
) -> float:
    """Check for a secondary eclipse near phase ≈ 0.5.

    Parameters
    ----------
    phase : np.ndarray
        Phase values in [-0.5, 0.5].
    flux : np.ndarray
        Corresponding flux values.

    Returns
    -------
    float
        Depth of the secondary eclipse (positive means a dip is present).
        Returns 0.0 if no significant secondary is found.
    """
    if len(phase) < 10:
        return 0.0

    # The secondary eclipse is expected near phase = ±0.5.  Because our
    # phase convention is [-0.5, 0.5], the secondary sits near |phase| ≈ 0.5.
    secondary_mask = np.abs(np.abs(phase) - 0.5) < 0.05
    baseline_mask = (np.abs(phase) > 0.1) & (np.abs(phase) < 0.4)

    if secondary_mask.sum() < 3 or baseline_mask.sum() < 5:
        return 0.0

    baseline = np.nanmedian(flux[baseline_mask])
    secondary_flux = np.nanmedian(flux[secondary_mask])

    depth = baseline - secondary_flux
    return float(max(depth, 0.0))



def _compute_snr(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    epoch: float,
    duration: float,
    depth: float,
) -> float:
    """Compute transit signal-to-noise ratio.

    SNR = depth × sqrt(n_in_transit × n_transits) / scatter

    Parameters
    ----------
    time, flux, flux_err : np.ndarray
        Cleaned arrays.
    period, epoch, duration, depth : float
        Transit parameters.

    Returns
    -------
    float
        Signal-to-noise ratio.  Returns 0.0 on failure.
    """
    if period <= 0 or duration <= 0 or depth <= 0 or len(time) < 5:
        return 0.0

    phase = phase_fold(time, period, epoch)
    phase_half_dur = (duration / 2.0) / period

    in_transit = np.abs(phase) <= phase_half_dur
    n_in_transit = in_transit.sum()

    if n_in_transit < 1:
        return 0.0

    # Number of distinct transits observed
    transit_numbers = np.round((time - epoch) / period)
    n_transits = len(np.unique(transit_numbers[in_transit]))

    if n_transits < 1:
        n_transits = 1

    # Out-of-transit scatter
    oot = ~in_transit
    if oot.sum() < 3:
        # Fall back to flux_err
        scatter = np.nanmedian(flux_err)
    else:
        scatter = np.nanstd(flux[oot])

    if scatter <= 0:
        scatter = np.nanmedian(flux_err) if np.nanmedian(flux_err) > 0 else 1.0

    snr = depth * np.sqrt(n_in_transit * n_transits) / scatter
    return float(snr)



@timer
def prepare_training_data(
    preprocessed_dir: Path,
    catalog_df: pd.DataFrame,
    detection_results_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a complete training dataset for the classifier.

    Iterates over preprocessed light curves, runs feature extraction for
    each detected signal, and constructs:

    * **X_global** – global phase-folded views, shape ``(N, 201)``
    * **X_local** – local transit views, shape ``(N, 61)``
    * **X_features** – numerical feature vectors, shape ``(N, 9)``
    * **y_labels** – integer class labels, shape ``(N,)``

    Parameters
    ----------
    preprocessed_dir : Path
        Directory containing preprocessed light-curve pickle files.  Each
        pickle is expected to be a dict with keys ``time``, ``flux``,
        ``flux_err``, and ``tic_id``.
    catalog_df : pd.DataFrame
        Catalog with columns ``tic_id`` and ``label`` (string class names
        matching ``config.CLASSIFICATION_CLASSES``).
    detection_results_df : pd.DataFrame
        DataFrame of signal-detection results with at least columns:
        ``tic_id``, ``period``, ``epoch`` (or ``t0``), ``duration``,
        ``depth``, ``sde``.

    Returns
    -------
    tuple of np.ndarray
        ``(X_global, X_local, X_features, y_labels)``
    """
    from src.utils import ProgressTracker, load_pickle

    preprocessed_dir = Path(preprocessed_dir)
    lc_files = sorted(preprocessed_dir.glob("*.pkl"))

    if len(lc_files) == 0:
        logger.warning("prepare_training_data: no pickle files found in %s", preprocessed_dir)
        return (
            np.empty((0, config.GLOBAL_VIEW_BINS)),
            np.empty((0, config.LOCAL_VIEW_BINS)),
            np.empty((0, 9)),
            np.empty(0, dtype=int),
        )

    # Build lookup maps
    label_map = {
        name: idx for idx, name in enumerate(config.CLASSIFICATION_CLASSES)
    }
    catalog_lookup = (
        catalog_df.set_index("tic_id")["label"].to_dict()
        if "tic_id" in catalog_df.columns and "label" in catalog_df.columns
        else {}
    )

    # Index detection results by tic_id for fast lookup
    detection_lookup: Dict[int, pd.Series] = {}
    if "tic_id" in detection_results_df.columns:
        for _, row in detection_results_df.iterrows():
            detection_lookup[int(row["tic_id"])] = row

    feature_names = [
        "transit_depth", "transit_duration", "period", "sde", "snr",
        "v_shape", "odd_even_ratio", "secondary_depth", "duration_period_ratio",
    ]

    global_views: List[np.ndarray] = []
    local_views: List[np.ndarray] = []
    feature_vecs: List[np.ndarray] = []
    labels: List[int] = []

    tracker = ProgressTracker(len(lc_files), description="Feature extraction")

    for lc_file in lc_files:
        try:
            lc_data = load_pickle(lc_file)
            from src.utils import tic_id_from_filename
            extracted_id = tic_id_from_filename(lc_file.name)
            if extracted_id is None:
                continue
            tic_id = int(extracted_id)
            # Skip if no detection result or no label
            if tic_id not in detection_lookup or tic_id not in catalog_lookup:
                tracker.update()
                continue

            det = detection_lookup[tic_id]
            label_str = catalog_lookup[tic_id]
            if label_str not in label_map:
                tracker.update()
                continue

            time_arr = np.asarray(lc_data["time"], dtype=np.float64)
            flux_arr = np.asarray(lc_data["flux"], dtype=np.float64)
            flux_err_arr = np.asarray(
                lc_data.get("flux_err", np.ones_like(flux_arr) * np.nanstd(flux_arr)),
                dtype=np.float64,
            )

            det_dict = det.to_dict() if hasattr(det, "to_dict") else dict(det)

            period = float(det_dict.get("period", 0))
            epoch = float(det_dict.get("epoch", det_dict.get("t0", 0)))
            duration = float(det_dict.get("duration", 0))

            if period <= 0 or duration <= 0:
                tracker.update()
                continue

            # --- Global view ---
            gv = create_global_view(time_arr, flux_arr, period, epoch)
            # --- Local view ---
            lv = create_local_view(time_arr, flux_arr, period, epoch, duration)
            # --- Numerical features ---
            feat_dict = extract_features(time_arr, flux_arr, flux_err_arr, det_dict)
            feat_vec = np.array([feat_dict[k] for k in feature_names], dtype=np.float64)

            global_views.append(gv)
            local_views.append(lv)
            feature_vecs.append(feat_vec)
            labels.append(label_map[label_str])

        except Exception as exc:
            logger.error("Error processing %s: %s", lc_file.name, exc)

        tracker.update()

    tracker.finish()

    n_samples = len(labels)
    logger.info(
        "Feature extraction complete: %d samples across %d classes",
        n_samples, len(set(labels)),
    )

    if n_samples == 0:
        return (
            np.empty((0, config.GLOBAL_VIEW_BINS)),
            np.empty((0, config.LOCAL_VIEW_BINS)),
            np.empty((0, 9)),
            np.empty(0, dtype=int),
        )

    X_global = np.vstack(global_views)       # (N, 201)
    X_local = np.vstack(local_views)         # (N, 61)
    X_features = np.vstack(feature_vecs)     # (N, 9)
    y_labels = np.array(labels, dtype=int)   # (N,)

    # Replace any remaining NaN/inf with 0 for safety
    X_global = np.nan_to_num(X_global, nan=0.0, posinf=0.0, neginf=0.0)
    X_local = np.nan_to_num(X_local, nan=0.0, posinf=0.0, neginf=0.0)
    X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)

    return X_global, X_local, X_features, y_labels



def _interpolate_nans(arr: np.ndarray) -> np.ndarray:
    """Fill NaN values via linear interpolation; edge NaNs use nearest.

    Parameters
    ----------
    arr : np.ndarray
        1-D array potentially containing NaN values.

    Returns
    -------
    np.ndarray
        Array with NaN values filled by interpolation.
    """
    nans = np.isnan(arr)
    if not nans.any():
        return arr
    if nans.all():
        return np.zeros_like(arr)

    x = np.arange(len(arr))
    arr_filled = arr.copy()
    arr_filled[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return arr_filled
