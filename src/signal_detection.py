"""
Signal Detection Module — Exoplanet Detection Pipeline.

Implements Box Least Squares (BLS) and Transit Least Squares (TLS)
periodogram searches to identify periodic transit-like dips in light curves.
Supports iterative multi-signal detection with transit masking.
"""

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.utils import logger, timer



@dataclass
class DetectionResult:
    """Container for a single transit signal detection.

    Attributes:
        period:  Orbital period in days.
        epoch:   Mid-transit time (BJD).
        depth:   Fractional transit depth.
        duration: Transit duration in days.
        sde:     Signal Detection Efficiency.
        fap:     Approximate False Alarm Probability.
        power:   Full periodogram power spectrum.
        periods: Period grid over which the search was performed.
        method:  Detection method used ('bls' or 'tls').
    """
    period: float
    epoch: float
    depth: float
    duration: float
    sde: float
    fap: float
    power: np.ndarray
    periods: np.ndarray
    method: str



def compute_sde(power: np.ndarray) -> float:
    """Compute Signal Detection Efficiency from periodogram power.

    SDE measures how many standard deviations the peak power exceeds
    the mean, providing a normalised detection metric.

    Args:
        power: 1-D array of periodogram power values.

    Returns:
        SDE value (float).  Returns 0.0 if the standard deviation is
        zero (e.g. flat power spectrum).
    """
    power = np.asarray(power, dtype=np.float64)
    if power.size == 0:
        return 0.0

    mean_power = np.nanmean(power)
    std_power = np.nanstd(power)

    if std_power == 0.0:
        return 0.0

    return float((np.nanmax(power) - mean_power) / std_power)


def compute_fap(sde: float) -> float:
    """Approximate False Alarm Probability from SDE using Gaussian tail.

    Uses the complementary error function (erfc) to estimate the
    probability that a signal with the given SDE arises from noise.

    Args:
        sde: Signal Detection Efficiency value.

    Returns:
        Approximate FAP in the range [0, 1].
    """
    from scipy.special import erfc

    if sde <= 0.0:
        return 1.0

    # One-sided Gaussian survival function: P(X > sde) = 0.5 * erfc(sde / sqrt(2))
    fap = 0.5 * erfc(sde / np.sqrt(2.0))
    return float(fap)



@timer
def run_bls(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
) -> DetectionResult:
    """Run a Box Least Squares periodogram search.

    Uses ``astropy.timeseries.BoxLeastSquares`` to search for box-shaped
    (flat-bottomed) transit signals across a grid of trial periods and
    durations.

    Args:
        time:     1-D array of observation times (BJD).
        flux:     1-D array of normalised flux values.
        flux_err: Optional 1-D array of flux uncertainties.

    Returns:
        Tuple of (period, epoch, depth, duration, sde, results) where
        *results* is the full ``BoxLeastSquareResults`` object from Astropy.

    Raises:
        ValueError: If input arrays are too short for a meaningful search.
    """
    from astropy.timeseries import BoxLeastSquares
    import astropy.units as u

    # --- Input validation ------------------------------------------------
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    finite_mask = np.isfinite(time) & np.isfinite(flux)
    if flux_err is not None:
        flux_err = np.asarray(flux_err, dtype=np.float64)
        finite_mask &= np.isfinite(flux_err)
        flux_err = flux_err[finite_mask]
    else:
        flux_err = None

    time = time[finite_mask]
    flux = flux[finite_mask]

    if len(time) < 50:
        raise ValueError(
            f"Insufficient data points for BLS search: {len(time)} "
            "(need ≥ 50)."
        )

    # --- Build BLS model -------------------------------------------------
    if flux_err is not None:
        model = BoxLeastSquares(time * u.day, flux, dy=flux_err)
    else:
        model = BoxLeastSquares(time * u.day, flux)

    # Period grid
    periods = np.linspace(
        config.PERIOD_MIN, config.PERIOD_MAX, config.PERIOD_GRID_SIZE
    ) * u.day

    # Duration grid
    durations = np.linspace(
        config.DURATION_MIN, config.DURATION_MAX, config.N_DURATIONS
    ) * u.day

    # --- Run periodogram -------------------------------------------------
    logger.info(
        f"BLS: searching {config.PERIOD_GRID_SIZE} periods "
        f"[{config.PERIOD_MIN}-{config.PERIOD_MAX} d], "
        f"{config.N_DURATIONS} durations "
        f"[{config.DURATION_MIN}-{config.DURATION_MAX} d]"
    )

    results = model.power(periods, durations)

    # --- Extract best-fit parameters -------------------------------------
    power = np.asarray(results.power)
    sde = compute_sde(power)

    idx_best = np.nanargmax(power)
    best_period = float(results.period[idx_best].value)
    best_duration = float(results.duration[idx_best].value)

    # Compute transit epoch (time of first transit)
    best_t0 = float(results.transit_time[idx_best].value)

    # Compute transit depth
    best_depth = float(results.depth[idx_best])

    logger.info(
        f"BLS result: P={best_period:.5f} d, t0={best_t0:.5f}, "
        f"depth={best_depth:.6f}, dur={best_duration:.4f} d, SDE={sde:.2f}"
    )

    fap = compute_fap(sde)
    power_arr = np.asarray(results.power)
    period_arr = np.asarray(
        [p.value if hasattr(p, "value") else p for p in results.period]
    )

    return DetectionResult(
        period=best_period,
        epoch=best_t0,
        depth=best_depth,
        duration=best_duration,
        sde=sde,
        fap=fap,
        power=power_arr,
        periods=period_arr,
        method="bls",
    )



@timer
def run_tls(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
) -> DetectionResult:
    """Run a Transit Least Squares periodogram search.

    Uses the ``transitleastsquares`` library which fits realistic
    limb-darkened transit models and is generally more sensitive than
    BLS for detecting shallow planetary transits.

    Args:
        time:     1-D array of observation times (BJD).
        flux:     1-D array of normalised flux values.
        flux_err: Optional 1-D array of flux uncertainties.

    Returns:
        Tuple of (period, epoch, depth, duration, sde, results) where
        *results* is the full TLS results object.

    Raises:
        ValueError: If input arrays are too short for a meaningful search.
        ImportError: If ``transitleastsquares`` is not installed.
    """
    try:
        from transitleastsquares import transitleastsquares as TLS
    except ImportError:
        raise ImportError(
            "The 'transitleastsquares' package is required for TLS. "
            "Install it with: pip install transitleastsquares"
        )

    # --- Input validation ------------------------------------------------
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    finite_mask = np.isfinite(time) & np.isfinite(flux)
    if flux_err is not None:
        flux_err = np.asarray(flux_err, dtype=np.float64)
        finite_mask &= np.isfinite(flux_err)
        flux_err = flux_err[finite_mask]
    else:
        flux_err = None

    time = time[finite_mask]
    flux = flux[finite_mask]

    if len(time) < 50:
        raise ValueError(
            f"Insufficient data points for TLS search: {len(time)} "
            "(need ≥ 50)."
        )

    # --- Build and run TLS -----------------------------------------------
    logger.info(
        f"TLS: searching periods [{config.PERIOD_MIN}-{config.PERIOD_MAX} d]"
    )

    model = TLS(time, flux, flux_err)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = model.power(
            period_min=config.PERIOD_MIN,
            period_max=config.PERIOD_MAX,
            n_transits_min=2,
            show_progress_bar=False,
        )

    # --- Extract best-fit parameters -------------------------------------
    best_period = float(results.period)
    best_t0 = float(results.T0)
    best_depth = float(1.0 - results.depth) if hasattr(results, "depth") else 0.0
    best_duration = float(results.duration)

    # TLS provides its own SDE
    sde = float(results.SDE) if hasattr(results, "SDE") else 0.0

    logger.info(
        f"TLS result: P={best_period:.5f} d, t0={best_t0:.5f}, "
        f"depth={best_depth:.6f}, dur={best_duration:.4f} d, SDE={sde:.2f}"
    )

    fap = compute_fap(sde)
    power_arr = np.asarray(getattr(results, "power", np.array([])))
    period_arr = np.asarray(getattr(results, "periods", np.array([])))

    return DetectionResult(
        period=best_period,
        epoch=best_t0,
        depth=best_depth,
        duration=best_duration,
        sde=sde,
        fap=fap,
        power=power_arr,
        periods=period_arr,
        method="tls",
    )



def mask_transit(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    duration: float,
    mask_width: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove data points that fall within detected transits.

    Masks out a window of width ``mask_width * duration`` centred on each
    predicted transit mid-time, enabling iterative multi-signal detection.

    Args:
        time:       1-D array of observation times.
        flux:       1-D array of flux values (same length as *time*).
        period:     Orbital period of the transit to mask (days).
        epoch:      Mid-transit reference epoch (BJD).
        duration:   Transit duration (days).
        mask_width: Multiplicative factor for the masking window
                    (default 1.5 × duration).

    Returns:
        Tuple of (masked_time, masked_flux) with transit points removed.
    """
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    if period <= 0:
        logger.warning("mask_transit: non-positive period; returning original data.")
        return time.copy(), flux.copy()

    # Phase-fold to identify in-transit points
    phase = ((time - epoch + 0.5 * period) % period) - 0.5 * period
    half_mask = 0.5 * mask_width * duration
    in_transit = np.abs(phase) < half_mask

    n_masked = int(np.sum(in_transit))
    logger.debug(
        f"mask_transit: masking {n_masked}/{len(time)} points "
        f"(P={period:.4f} d, width={mask_width:.1f}×{duration:.4f} d)"
    )

    keep = ~in_transit
    return time[keep], flux[keep]



@timer
def detect_signals(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    method: str = "bls",
    max_signals: int = 3,
) -> List[DetectionResult]:
    """Iteratively detect multiple periodic transit signals.

    Searches for the strongest signal, masks it, and repeats up to
    *max_signals* times or until the SDE drops below the configured
    threshold (``config.SDE_THRESHOLD``).

    Args:
        time:        1-D array of observation times (BJD).
        flux:        1-D array of normalised flux values.
        flux_err:    Optional 1-D array of flux uncertainties.
        method:      Detection method — ``'bls'`` or ``'tls'``.
        max_signals: Maximum number of signals to search for.

    Returns:
        List of :class:`DetectionResult` objects, one per detected signal,
        ordered by detection (strongest first).

    Raises:
        ValueError: If *method* is not ``'bls'`` or ``'tls'``.
    """
    method = method.lower().strip()
    if method not in ("bls", "tls"):
        raise ValueError(f"Unknown detection method '{method}'. Use 'bls' or 'tls'.")

    search_fn = run_tls if method == "tls" else run_bls

    # Work on copies to avoid mutating the caller's arrays
    t_work = np.asarray(time, dtype=np.float64).copy()
    f_work = np.asarray(flux, dtype=np.float64).copy()
    ferr_work: Optional[np.ndarray] = None
    if flux_err is not None:
        ferr_work = np.asarray(flux_err, dtype=np.float64).copy()

    detections: List[DetectionResult] = []

    for i in range(max_signals):
        logger.info(f"Signal search iteration {i + 1}/{max_signals}")

        if len(t_work) < 50:
            logger.warning(
                f"Only {len(t_work)} data points remain -- stopping search."
            )
            break

        try:
            detection = search_fn(t_work, f_work, ferr_work)
        except Exception as exc:
            logger.error(f"Signal search failed on iteration {i + 1}: {exc}")
            break

        # Check SDE threshold
        if detection.sde < config.SDE_THRESHOLD:
            logger.info(
                f"SDE {detection.sde:.2f} < threshold {config.SDE_THRESHOLD} -- "
                "stopping search."
            )
            break

        detections.append(detection)

        logger.info(
            f"Signal {i + 1} detected: P={detection.period:.5f} d, "
            f"SDE={detection.sde:.2f}, FAP={detection.fap:.2e}"
        )

        # Mask detected transit before next iteration
        if detection.duration <= 0:
            logger.warning("Non-positive duration -- cannot mask transit.")
            break

        t_work, f_work = mask_transit(t_work, f_work, detection.period,
                                       detection.epoch, detection.duration)

        # Also mask flux_err if present
        if ferr_work is not None:
            # Recompute mask for flux_err (same logic as mask_transit)
            remaining_indices = np.isin(
                np.asarray(time, dtype=np.float64), t_work
            )
            ferr_work = np.asarray(flux_err, dtype=np.float64)[remaining_indices]
            # Keep only as many elements as t_work (handle floating-point matching)
            if len(ferr_work) != len(t_work):
                # Fall-back: reindex using the mask_transit logic directly
                ferr_work = _mask_array(
                    np.asarray(time, dtype=np.float64),
                    np.asarray(flux_err, dtype=np.float64),
                    detections,
                )

    logger.info(f"Detection complete: {len(detections)} signal(s) found.")
    return detections


def _mask_array(
    time: np.ndarray,
    arr: np.ndarray,
    detections: List[DetectionResult],
    mask_width: float = 1.5,
) -> np.ndarray:
    """Cumulatively mask *arr* for all detected transits.

    Internal helper used to keep ``flux_err`` in sync when iterating.
    """
    keep = np.ones(len(time), dtype=bool)
    for det in detections:
        if det.period <= 0 or det.duration <= 0:
            continue
        phase = ((time - det.epoch + 0.5 * det.period) % det.period) - 0.5 * det.period
        half_mask = 0.5 * mask_width * det.duration
        keep &= np.abs(phase) >= half_mask
    return arr[keep]
