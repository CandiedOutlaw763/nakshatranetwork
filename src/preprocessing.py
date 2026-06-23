"""
Preprocessing module for the Exoplanet Detection Pipeline.

Handles cleaning and preparing TESS light curves for downstream analysis:
  - Loading raw FITS / pickle light curves
  - Quality-flag filtering
  - Sigma-clipping outlier removal
  - Median normalization with error propagation
  - Detrending (biweight, Savitzky-Golay, lightkurve flatten)
  - Gap handling (interpolation / segment splitting)
  - Batch parallel preprocessing
"""

import sys
import pickle
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from multiprocessing import Pool

import numpy as np
from scipy.signal import savgol_filter

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.utils import logger, timer, sigma_clip, normalize_flux, save_pickle, load_pickle



def load_lightcurve(
    filepath: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a raw light curve from a FITS or pickle file.

    Supports:
      - FITS files produced by the SPOC pipeline (via ``astropy.io.fits``).
      - Pickle files previously saved by this pipeline.

    Parameters
    ----------
    filepath : str or Path
        Path to the light-curve file (.fits or .pkl / .pickle).

    Returns
    -------
    time : np.ndarray
        Observation timestamps (BJD - 2457000 for TESS).
    flux : np.ndarray
        Flux values (``pdcsap_flux`` by default for FITS files).
    flux_err : np.ndarray
        Flux uncertainty values.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the file extension is unsupported or required columns are missing.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Light-curve file not found: {filepath}")

    suffix = filepath.suffix.lower()

    # ------------------------------------------------------------------
    # FITS format
    # ------------------------------------------------------------------
    if suffix in (".fits", ".fit"):
        from astropy.io import fits

        with fits.open(filepath, mode="readonly", memmap=True) as hdul:
            data = hdul[1].data  # Binary table extension

            time = np.asarray(data["TIME"], dtype=np.float64)
            col_names_upper = [c.name.upper() for c in data.columns]
            
            flux_col = config.FLUX_COLUMN.upper()
            if flux_col not in col_names_upper:
                if "FLUX" in col_names_upper:
                    flux_col = "FLUX"
                else:
                    raise ValueError(
                        f"Flux column '{config.FLUX_COLUMN}' or 'FLUX' not found in FITS. "
                        f"Available: {[c.name for c in data.columns]}"
                    )
            
            err_col = flux_col + "_ERR"
            
            # The actual column name might be lowercase in data
            actual_flux_col = [c.name for c in data.columns if c.name.upper() == flux_col][0]
            flux = np.asarray(data[actual_flux_col], dtype=np.float64)

            # Error column may be missing in some products
            if err_col in col_names_upper:
                actual_err_col = [c.name for c in data.columns if c.name.upper() == err_col][0]
                flux_err = np.asarray(data[actual_err_col], dtype=np.float64)
            else:
                flux_err = np.full_like(flux, np.nan)
                logger.warning(
                    "Flux-error column '%s' not found – filled with NaN.", err_col
                )

            # Also grab quality flags if present (stored for later use)
            if "QUALITY" in [c.name.upper() for c in data.columns]:
                quality = np.asarray(data["QUALITY"], dtype=np.int32)
                # Attach as metadata so callers can optionally use it
                time = np.asarray(time)
                time.flags.writeable = True

        logger.info("Loaded FITS light curve: %s (%d points)", filepath.name, len(time))
        return time, flux, flux_err

    # ------------------------------------------------------------------
    # Pickle format
    # ------------------------------------------------------------------
    if suffix in (".pkl", ".pickle"):
        obj = load_pickle(filepath)

        # Accept dict with 'time', 'flux', 'flux_err' keys …
        if isinstance(obj, dict):
            time = np.asarray(obj["time"], dtype=np.float64)
            flux = np.asarray(obj["flux"], dtype=np.float64)
            flux_err = np.asarray(
                obj.get("flux_err", np.full_like(flux, np.nan)),
                dtype=np.float64,
            )
        # … or a tuple / list of arrays
        elif isinstance(obj, (tuple, list)):
            if len(obj) < 2:
                raise ValueError(
                    "Pickle must contain at least (time, flux); got "
                    f"{len(obj)} element(s)."
                )
            time = np.asarray(obj[0], dtype=np.float64)
            flux = np.asarray(obj[1], dtype=np.float64)
            flux_err = (
                np.asarray(obj[2], dtype=np.float64)
                if len(obj) > 2
                else np.full_like(flux, np.nan)
            )
        else:
            raise ValueError(
                f"Unsupported pickle content type: {type(obj).__name__}. "
                "Expected dict or tuple/list of arrays."
            )

        logger.info(
            "Loaded pickle light curve: %s (%d points)", filepath.name, len(time)
        )
        return time, flux, flux_err

    raise ValueError(f"Unsupported file extension: '{suffix}'")



def remove_bad_quality(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    quality: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter out data points with non-zero quality flags.

    Parameters
    ----------
    time, flux, flux_err : np.ndarray
        Light-curve arrays (same length).
    quality : np.ndarray
        Integer quality-flag array from the TESS FITS file.  Points with
        ``quality != 0`` are discarded.

    Returns
    -------
    time, flux, flux_err : tuple of np.ndarray
        Filtered arrays.
    """
    if len(time) == 0:
        logger.warning("remove_bad_quality: received empty arrays.")
        return time, flux, flux_err

    mask = quality == 0
    n_removed = int((~mask).sum())
    if n_removed > 0:
        logger.info(
            "Removed %d/%d points with non-zero quality flags.",
            n_removed,
            len(time),
        )

    return time[mask], flux[mask], flux_err[mask]



def remove_outliers(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    sigma: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iterative sigma-clipping to remove outlier flux points.

    Uses the ``sigma_clip`` utility from ``src.utils`` which iterates
    until convergence or a maximum of 5 iterations.

    Parameters
    ----------
    time, flux, flux_err : np.ndarray
        Light-curve arrays.
    sigma : float, optional
        Number of standard deviations for clipping (default from config).

    Returns
    -------
    time, flux, flux_err : tuple of np.ndarray
        Arrays with outliers removed.
    """
    if len(flux) == 0:
        logger.warning("remove_outliers: received empty flux array.")
        return time, flux, flux_err

    # Remove NaN / Inf values first
    finite_mask = np.isfinite(flux)
    if not finite_mask.any():
        logger.warning("remove_outliers: all flux values are non-finite.")
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    # Sigma-clip on finite data
    good = sigma_clip(flux, sigma=sigma)
    combined_mask = finite_mask & good

    n_removed = int((~combined_mask).sum())
    if n_removed > 0:
        logger.info(
            "Sigma-clip (sigma=%.1f): removed %d/%d outlier points.",
            sigma,
            n_removed,
            len(flux),
        )

    return time[combined_mask], flux[combined_mask], flux_err[combined_mask]



def normalize(
    flux: np.ndarray,
    flux_err: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Median-normalize the flux and propagate uncertainties.

    Each value is divided by the median flux so the baseline sits at ~1.0.
    The uncertainty is scaled by the same factor.

    Parameters
    ----------
    flux : np.ndarray
        Raw or partially-processed flux values.
    flux_err : np.ndarray
        Corresponding flux uncertainties.

    Returns
    -------
    norm_flux : np.ndarray
        Normalized flux (median ≈ 1.0).
    norm_flux_err : np.ndarray
        Propagated normalized uncertainty.
    """
    if len(flux) == 0:
        logger.warning("normalize: received empty flux array.")
        return flux.copy(), flux_err.copy()

    median = np.nanmedian(flux)
    if median == 0 or not np.isfinite(median):
        logger.warning(
            "normalize: median flux is %s – returning un-normalized arrays.",
            median,
        )
        return flux.copy(), flux_err.copy()

    norm_flux = flux / median
    norm_flux_err = flux_err / median

    logger.debug("Normalized flux by median = %.4f", median)
    return norm_flux, norm_flux_err



def detrend(
    time: np.ndarray,
    flux: np.ndarray,
    method: str = "biweight",
    window_length: int = 401,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove long-term stellar variability from the flux.

    Parameters
    ----------
    time : np.ndarray
        Observation timestamps.
    flux : np.ndarray
        Flux values (ideally already normalized to ~1).
    method : {'biweight', 'savgol', 'flatten'}
        Detrending strategy:
        - ``'biweight'``: Wotan's time-windowed biweight filter.
        - ``'savgol'``: Scipy Savitzky-Golay polynomial filter.
        - ``'flatten'``: Lightkurve's built-in flatten method.
    window_length : int, optional
        Window width in cadences (must be odd for ``savgol``).

    Returns
    -------
    detrended_flux : np.ndarray
        Flux with the trend divided out.
    trend : np.ndarray
        The estimated trend that was removed.

    Raises
    ------
    ValueError
        If *method* is not recognised.
    """
    if len(flux) == 0:
        logger.warning("detrend: received empty arrays.")
        return flux.copy(), np.array([], dtype=np.float64)

    method = method.lower()

    # ------------------------------------------------------------------
    # Biweight (wotan)
    # ------------------------------------------------------------------
    if method == "biweight":
        try:
            from wotan import flatten as wotan_flatten

            # wotan expects window_length in units of *time* (days).
            # Convert cadences → days using median cadence spacing.
            dt = np.nanmedian(np.diff(time))
            window_days = window_length * dt

            detrended_flux, trend = wotan_flatten(
                time,
                flux,
                method="biweight",
                window_length=window_days,
                return_trend=True,
            )
            logger.info(
                "Detrended with biweight filter (window=%.2f d).", window_days
            )
        except ImportError:
            logger.warning(
                "wotan not installed – falling back to Savitzky-Golay detrending."
            )
            return detrend(time, flux, method="savgol", window_length=window_length)

    # ------------------------------------------------------------------
    # Savitzky-Golay
    # ------------------------------------------------------------------
    elif method == "savgol":
        wl = window_length if window_length % 2 == 1 else window_length + 1
        # Handle arrays shorter than the window
        if len(flux) < wl:
            wl = len(flux) if len(flux) % 2 == 1 else len(flux) - 1
        if wl < 3:
            logger.warning(
                "detrend(savgol): too few points (%d) to apply filter.", len(flux)
            )
            return flux.copy(), np.ones_like(flux)

        polyorder = min(3, wl - 1)
        trend = savgol_filter(flux, window_length=wl, polyorder=polyorder)
        # Avoid division by zero in the trend
        trend_safe = np.where(trend == 0, 1.0, trend)
        detrended_flux = flux / trend_safe
        logger.info(
            "Detrended with Savitzky-Golay filter (window=%d, poly=%d).",
            wl,
            polyorder,
        )

    # ------------------------------------------------------------------
    # Lightkurve flatten
    # ------------------------------------------------------------------
    elif method == "flatten":
        try:
            import lightkurve as lk

            lc = lk.LightCurve(time=time, flux=flux)
            flat_lc, trend_lc = lc.flatten(
                window_length=window_length, return_trend=True
            )
            detrended_flux = flat_lc.flux.value
            trend = trend_lc.flux.value
            logger.info(
                "Detrended with lightkurve flatten (window=%d).", window_length
            )
        except ImportError:
            logger.warning(
                "lightkurve not installed – falling back to Savitzky-Golay."
            )
            return detrend(time, flux, method="savgol", window_length=window_length)

    else:
        raise ValueError(
            f"Unknown detrend method '{method}'. "
            "Choose from 'biweight', 'savgol', or 'flatten'."
        )

    return detrended_flux, trend



def fill_gaps(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    max_gap: float = 0.5,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
]:
    """Handle gaps in a time series.

    Small gaps (≤ *max_gap* days) are filled by linear interpolation.
    Large gaps (> *max_gap*) cause the light curve to be split into
    independent segments.

    Parameters
    ----------
    time, flux, flux_err : np.ndarray
        Light-curve arrays (same length, sorted by time).
    max_gap : float, optional
        Maximum gap (in days) to interpolate across.  Gaps larger than this
        trigger a segment split.

    Returns
    -------
    result : tuple or list of tuples
        If the light curve is contiguous (no large gaps), returns a single
        ``(time, flux, flux_err)`` tuple with small gaps interpolated.
        If large gaps exist, returns a **list** of per-segment tuples.
    """
    if len(time) == 0:
        logger.warning("fill_gaps: received empty arrays.")
        return time, flux, flux_err

    # Sort by time for safety
    sort_idx = np.argsort(time)
    time = time[sort_idx]
    flux = flux[sort_idx]
    flux_err = flux_err[sort_idx]

    dt = np.diff(time)
    median_cadence = np.nanmedian(dt)

    # Identify large-gap positions
    large_gap_idx = np.where(dt > max_gap)[0]

    if len(large_gap_idx) == 0:
        # No large gaps – interpolate any NaN / small internal gaps
        time, flux, flux_err = _interpolate_small_gaps(
            time, flux, flux_err, median_cadence
        )
        return time, flux, flux_err

    # Split into segments at large gaps
    split_points = np.concatenate([[0], large_gap_idx + 1, [len(time)]])
    segments: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for i in range(len(split_points) - 1):
        s, e = split_points[i], split_points[i + 1]
        seg_t, seg_f, seg_e = (
            time[s:e].copy(),
            flux[s:e].copy(),
            flux_err[s:e].copy(),
        )
        if len(seg_t) < 3:
            continue  # Discard tiny segments

        seg_t, seg_f, seg_e = _interpolate_small_gaps(
            seg_t, seg_f, seg_e, median_cadence
        )
        segments.append((seg_t, seg_f, seg_e))

    logger.info(
        "Split light curve into %d segments (max_gap=%.2f d).",
        len(segments),
        max_gap,
    )

    if len(segments) == 1:
        return segments[0]
    return segments


def _interpolate_small_gaps(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    cadence: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearly interpolate NaN values and small internal cadence gaps.

    Parameters
    ----------
    time, flux, flux_err : np.ndarray
        Light-curve segment.
    cadence : float
        Median cadence spacing (days).

    Returns
    -------
    time, flux, flux_err : tuple of np.ndarray
        Arrays with NaN gaps filled by interpolation.
    """
    nan_mask = ~np.isfinite(flux)
    if nan_mask.sum() == 0:
        return time, flux, flux_err

    if nan_mask.all():
        logger.warning("_interpolate_small_gaps: entire segment is NaN.")
        return time, flux, flux_err

    good = ~nan_mask
    flux[nan_mask] = np.interp(time[nan_mask], time[good], flux[good])

    # Interpolated flux_err: use mean of neighbouring errors
    if np.isfinite(flux_err[good]).any():
        flux_err[nan_mask] = np.interp(
            time[nan_mask], time[good], flux_err[good]
        )
    else:
        flux_err[nan_mask] = np.nanmedian(flux_err)

    return time, flux, flux_err



@timer
def preprocess_lightcurve(
    filepath: Union[str, Path],
) -> Dict[str, np.ndarray]:
    """Run the complete preprocessing pipeline on a single light curve.

    Steps executed in order:
      1. Load from FITS / pickle
      2. Remove non-finite values
      3. Remove outliers (sigma-clip)
      4. Normalize (median)
      5. Detrend (method from config)
      6. Fill gaps / split segments

    Parameters
    ----------
    filepath : str or Path
        Path to the raw light-curve file.

    Returns
    -------
    result : dict
        Dictionary containing:
        - ``'time'``: cleaned time array
        - ``'flux'``: cleaned, detrended flux
        - ``'flux_err'``: propagated uncertainties
        - ``'trend'``: estimated stellar trend
        - ``'segments'``: list of segments if large gaps exist, else ``None``
        - ``'filepath'``: original file path
    """
    filepath = Path(filepath)

    # 1. Load
    time, flux, flux_err = load_lightcurve(filepath)

    # 2. Remove non-finite points
    finite_mask = np.isfinite(time) & np.isfinite(flux)
    time, flux, flux_err = time[finite_mask], flux[finite_mask], flux_err[finite_mask]

    if len(time) == 0:
        logger.warning("No valid data points in %s after NaN removal.", filepath.name)
        return _empty_result(filepath)

    # 3. Remove outliers
    time, flux, flux_err = remove_outliers(
        time, flux, flux_err, sigma=config.SIGMA_CLIP_THRESHOLD
    )

    if len(time) == 0:
        logger.warning("No data left in %s after outlier removal.", filepath.name)
        return _empty_result(filepath)

    # 4. Normalize
    flux, flux_err = normalize(flux, flux_err)

    # 5. Detrend
    detrended, trend = detrend(
        time,
        flux,
        method=config.DETREND_METHOD,
        window_length=config.FLATTEN_WINDOW_LENGTH,
    )

    # 6. Fill gaps
    gap_result = fill_gaps(time, detrended, flux_err)

    segments = None
    if isinstance(gap_result, list):
        segments = gap_result
        # Use the longest segment as the primary arrays
        longest = max(segments, key=lambda s: len(s[0]))
        time, detrended, flux_err = longest
    else:
        time, detrended, flux_err = gap_result

    return {
        "time": time,
        "flux": detrended,
        "flux_err": flux_err,
        "trend": trend,
        "segments": segments,
        "filepath": str(filepath),
    }


def _empty_result(filepath: Path) -> Dict[str, np.ndarray]:
    """Return an empty result dictionary for a failed light curve."""
    return {
        "time": np.array([], dtype=np.float64),
        "flux": np.array([], dtype=np.float64),
        "flux_err": np.array([], dtype=np.float64),
        "trend": np.array([], dtype=np.float64),
        "segments": None,
        "filepath": str(filepath),
    }



def _process_and_save(args: Tuple[Path, Path]) -> Optional[str]:
    """Worker function for multiprocessing: preprocess + save one file.

    Parameters
    ----------
    args : tuple
        ``(input_path, output_path)`` pair.

    Returns
    -------
    str or None
        The output path on success, ``None`` on failure.
    """
    input_path, output_path = args
    try:
        result = preprocess_lightcurve(input_path)
        if len(result["time"]) == 0:
            logger.warning("Skipping %s – no data after preprocessing.", input_path.name)
            return None
        save_pickle(result, output_path)
        return str(output_path)
    except Exception as exc:
        logger.error("Failed to preprocess %s: %s", input_path.name, exc)
        return None


@timer
def preprocess_all(
    raw_dir: Union[str, Path],
    output_dir: Union[str, Path],
    n_jobs: int = 4,
) -> List[str]:
    """Batch-preprocess all light curves in *raw_dir* and save to *output_dir*.

    Supported input formats: ``.fits``, ``.fit``, ``.pkl``, ``.pickle``.
    Outputs are saved as pickle files (``<stem>_processed.pkl``).

    Parameters
    ----------
    raw_dir : str or Path
        Directory containing raw light-curve files.
    output_dir : str or Path
        Destination directory for processed pickle files.
    n_jobs : int, optional
        Number of parallel workers (default 4).

    Returns
    -------
    saved : list of str
        Paths of successfully processed and saved files.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect input files
    extensions = (".fits", ".fit", ".pkl", ".pickle")
    files = sorted(
        f for f in raw_dir.iterdir() if f.suffix.lower() in extensions
    )

    if not files:
        logger.warning("No light-curve files found in %s", raw_dir)
        return []

    logger.info("Found %d light-curve files in %s", len(files), raw_dir)

    # Prepare (input, output) pairs
    tasks = [
        (f, output_dir / f"{f.stem}_processed.pkl")
        for f in files
    ]

    # Run in parallel
    if n_jobs > 1 and len(tasks) > 1:
        with Pool(processes=min(n_jobs, len(tasks))) as pool:
            results = pool.map(_process_and_save, tasks)
    else:
        results = [_process_and_save(t) for t in tasks]

    saved = [r for r in results if r is not None]
    logger.info(
        "Batch preprocessing complete: %d/%d files processed successfully.",
        len(saved),
        len(files),
    )
    return saved



def preprocess_from_lightkurve(
    lc_object,
) -> Dict[str, np.ndarray]:
    """Preprocess a ``lightkurve.LightCurve`` object directly.

    Extracts ``time``, ``flux``, and ``flux_err`` from the object and runs
    the standard cleaning pipeline (outlier removal → normalization →
    detrending → gap filling).

    Parameters
    ----------
    lc_object : lightkurve.LightCurve
        A LightCurve object (e.g., from ``search_lightcurve(...).download()``).

    Returns
    -------
    result : dict
        Same structure as :func:`preprocess_lightcurve`.

    Raises
    ------
    TypeError
        If *lc_object* is not a lightkurve LightCurve instance.
    """
    try:
        import lightkurve as lk

        if not isinstance(lc_object, lk.LightCurve):
            raise TypeError(
                f"Expected a lightkurve.LightCurve, got {type(lc_object).__name__}."
            )
    except ImportError:
        logger.warning(
            "lightkurve not installed – attempting duck-typed attribute access."
        )

    # Extract arrays
    time = np.asarray(lc_object.time.value, dtype=np.float64)
    flux = np.asarray(lc_object.flux.value, dtype=np.float64)

    if hasattr(lc_object, "flux_err") and lc_object.flux_err is not None:
        flux_err = np.asarray(lc_object.flux_err.value, dtype=np.float64)
    else:
        flux_err = np.full_like(flux, np.nan)
        logger.warning("LightCurve object has no flux_err – filled with NaN.")

    # Remove non-finite
    finite_mask = np.isfinite(time) & np.isfinite(flux)
    time, flux, flux_err = time[finite_mask], flux[finite_mask], flux_err[finite_mask]

    if len(time) == 0:
        logger.warning("No valid points in lightkurve object after NaN removal.")
        return _empty_result(Path("lightkurve_object"))

    # Outlier removal
    time, flux, flux_err = remove_outliers(
        time, flux, flux_err, sigma=config.SIGMA_CLIP_THRESHOLD
    )

    if len(time) == 0:
        return _empty_result(Path("lightkurve_object"))

    # Normalize
    flux, flux_err = normalize(flux, flux_err)

    # Detrend
    detrended, trend = detrend(
        time,
        flux,
        method=config.DETREND_METHOD,
        window_length=config.FLATTEN_WINDOW_LENGTH,
    )

    # Fill gaps
    gap_result = fill_gaps(time, detrended, flux_err)

    segments = None
    if isinstance(gap_result, list):
        segments = gap_result
        longest = max(segments, key=lambda s: len(s[0]))
        time, detrended, flux_err = longest
    else:
        time, detrended, flux_err = gap_result

    return {
        "time": time,
        "flux": detrended,
        "flux_err": flux_err,
        "trend": trend,
        "segments": segments,
        "filepath": "lightkurve_object",
    }



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess TESS light curves for the exoplanet pipeline."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(config.RAW_DATA_DIR),
        help="Directory containing raw light-curve files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config.PROCESSED_DATA_DIR),
        help="Directory to save processed pickle files.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Number of parallel workers.",
    )
    args = parser.parse_args()

    saved = preprocess_all(args.raw_dir, args.output_dir, n_jobs=args.n_jobs)
    logger.info("Done. %d files saved to %s", len(saved), args.output_dir)
