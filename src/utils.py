"""
Shared utility functions for the Exoplanet Detection Pipeline.
Provides logging setup, data I/O, progress tracking, and common operations.
"""

import logging
import sys
import time
import pickle
import json
from pathlib import Path
from functools import wraps
from typing import Optional, Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config



def setup_logger(name: str = "exoplanet_pipeline",
                 level: str = None) -> logging.Logger:
    """Configure and return a logger with file + console handlers."""
    level = level or config.LOG_LEVEL
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(getattr(logging, level.upper()))

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console.setFormatter(console_fmt)
    logger.addHandler(console)

    # File handler
    file_handler = logging.FileHandler(config.LOG_FILE, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)-8s | %(funcName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()



def timer(func):
    """Decorator to log function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        logger.info(f"Starting: {func.__name__}")
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        if elapsed < 60:
            logger.info(f"Completed: {func.__name__} in {elapsed:.1f}s")
        else:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            logger.info(f"Completed: {func.__name__} in {minutes}m {seconds:.1f}s")
        return result
    return wrapper



def save_pickle(obj: Any, filepath: Path) -> None:
    """Save object as pickle file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.debug(f"Saved pickle: {filepath}")


def load_pickle(filepath: Path) -> Any:
    """Load object from pickle file."""
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    logger.debug(f"Loaded pickle: {filepath}")
    return obj


def save_json(data: dict, filepath: Path) -> None:
    """Save dictionary as JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=_json_serializer)
    logger.debug(f"Saved JSON: {filepath}")


def load_json(filepath: Path) -> dict:
    """Load dictionary from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def _json_serializer(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_results_csv(results: List[Dict], filepath: Path) -> None:
    """Save list of result dictionaries as CSV."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(results)} results to {filepath}")



def sigma_clip(data: np.ndarray, sigma: float = 5.0,
               maxiters: int = 5) -> np.ndarray:
    """Return boolean mask of non-outlier points (True = good)."""
    mask = np.ones(len(data), dtype=bool)
    for _ in range(maxiters):
        if mask.sum() < 3:
            break
        median = np.nanmedian(data[mask])
        std = np.nanstd(data[mask])
        if std == 0:
            break
        new_mask = np.abs(data - median) < sigma * std
        if np.array_equal(mask, new_mask):
            break
        mask = new_mask
    return mask


def normalize_flux(flux: np.ndarray, method: str = "median") -> np.ndarray:
    """Normalize flux array."""
    if method == "median":
        median = np.nanmedian(flux)
        if median != 0:
            return flux / median
        return flux
    elif method == "minmax":
        fmin, fmax = np.nanmin(flux), np.nanmax(flux)
        if fmax - fmin != 0:
            return (flux - fmin) / (fmax - fmin)
        return flux
    elif method == "standard":
        mean, std = np.nanmean(flux), np.nanstd(flux)
        if std != 0:
            return (flux - mean) / std
        return flux - mean
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def phase_fold(time: np.ndarray, period: float,
               epoch: float) -> np.ndarray:
    """Phase-fold time array at given period and epoch. Returns phases in [-0.5, 0.5]."""
    phase = ((time - epoch) / period) % 1.0
    phase[phase > 0.5] -= 1.0
    return phase


def bin_data(x: np.ndarray, y: np.ndarray, n_bins: int,
             x_range: Tuple[float, float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin y values by x into n_bins equally spaced bins.

    Returns:
        bin_centers, bin_means, bin_stds
    """
    if x_range is None:
        x_range = (np.nanmin(x), np.nanmax(x))

    bin_edges = np.linspace(x_range[0], x_range[1], n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_means = np.full(n_bins, np.nan)
    bin_stds = np.full(n_bins, np.nan)

    indices = np.digitize(x, bin_edges) - 1
    for i in range(n_bins):
        mask = indices == i
        if mask.sum() > 0:
            bin_means[i] = np.nanmean(y[mask])
            bin_stds[i] = np.nanstd(y[mask]) / np.sqrt(mask.sum()) if mask.sum() > 1 else 0.0

    return bin_centers, bin_means, bin_stds


def running_median(data: np.ndarray, window: int) -> np.ndarray:
    """Compute running median with given window size."""
    result = np.full_like(data, np.nan)
    half = window // 2
    for i in range(len(data)):
        start = max(0, i - half)
        end = min(len(data), i + half + 1)
        result[i] = np.nanmedian(data[start:end])
    return result



def tic_id_from_filename(filename: str) -> Optional[int]:
    """Extract TIC ID from a TESS light curve filename."""
    import re
    # Try custom format: TIC_12345_s0001.fits
    m1 = re.search(r"TIC_(\d+)_", str(filename))
    if m1:
        return int(m1.group(1))
        
    m2 = re.search(r"\[\'(\d+)\'\]", str(filename))
    if m2:
        return int(m2.group(1))
        
    # Format: tess2018206045859-s0001-0000000261136679-0120-s_lc.fits
    parts = str(filename).split("-")
    for part in parts:
        if len(part) >= 8 and part.isdigit():
            return int(part)
    return None


def match_catalogs(tic_ids: List[int],
                   toi_catalog: pd.DataFrame,
                   eb_catalog: pd.DataFrame) -> pd.DataFrame:
    """Cross-match TIC IDs with TOI and EB catalogs to create labels.

    Returns DataFrame with columns: tic_id, label, source
    """
    labels = []

    toi_tics = set(toi_catalog["tid"].values) if "tid" in toi_catalog.columns else set()
    eb_tics = set(eb_catalog["TIC"].values) if "TIC" in eb_catalog.columns else set()

    for tic_id in tic_ids:
        if tic_id in toi_tics:
            row = toi_catalog[toi_catalog["tid"] == tic_id].iloc[0]
            disp = row.get("tfopwg_disp", "")
            if disp in ("CP", "KP"):
                labels.append({"tic_id": tic_id, "label": "PLANET", "source": "TOI"})
            elif disp == "FP":
                labels.append({"tic_id": tic_id, "label": "BLEND", "source": "TOI"})
            elif disp == "PC":
                labels.append({"tic_id": tic_id, "label": "PLANET", "source": "TOI"})
            else:
                labels.append({"tic_id": tic_id, "label": "OTHER", "source": "TOI"})
        elif tic_id in eb_tics:
            labels.append({"tic_id": tic_id, "label": "ECLIPSING_BINARY", "source": "EB_CATALOG"})
        else:
            labels.append({"tic_id": tic_id, "label": "OTHER", "source": "NONE"})

    return pd.DataFrame(labels)



class ProgressTracker:
    """Simple progress tracker with ETA estimation."""

    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()

    def update(self, n: int = 1):
        self.current += n
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / rate if rate > 0 else 0

        pct = 100 * self.current / self.total
        bar_len = 30
        filled = int(bar_len * self.current / self.total)
        bar = "#" * filled + "-" * (bar_len - filled)

        sys.stdout.write(
            f"\r{self.description} |{bar}| {pct:5.1f}% "
            f"({self.current}/{self.total}) "
            f"ETA: {eta:.0f}s"
        )
        sys.stdout.flush()

        if self.current >= self.total:
            print()  # Newline at completion

    def finish(self):
        self.current = self.total
        self.update(0)
