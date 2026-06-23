"""
Data Acquisition Module for the Exoplanet Detection Pipeline.

Handles downloading TESS light curves, fetching exoplanet/EB catalogs,
creating labeled datasets via cross-matching, and generating synthetic
training data with the batman transit model.

Functions:
    download_tess_lightcurves  — Fetch 2-minute cadence TESS light curves for a sector.
    download_toi_catalog       — Retrieve the NASA Exoplanet Archive TOI catalog.
    download_eb_catalog        — Retrieve an eclipsing binary catalog (TESS-EBs / Vizier).
    create_labeled_dataset     — Cross-match TIC IDs with catalogs to assign labels.
    generate_synthetic_transits — Produce synthetic planet-transit light curves.
    generate_synthetic_ebs      — Produce synthetic eclipsing-binary light curves.
    run_full_acquisition        — Orchestrate the complete data-acquisition pipeline.
"""

import sys
import time
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project imports (config lives one level above src/)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.utils import logger, timer, save_pickle, save_results_csv

# Suppress noisy warnings from lightkurve / astroquery
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="astroquery")



TOI_CATALOG_URL: str = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
    "query=select+*+from+toi&format=csv"
)

TOI_KEY_COLUMNS: List[str] = [
    "toi", "tid", "tfopwg_disp", "sectors",
    "pl_orbper", "pl_trandep", "pl_trandur", "pl_rade",
]

MAX_DOWNLOAD_RETRIES: int = 3
RETRY_BACKOFF_SECONDS: float = 5.0



@timer
def query_sector_tic_ids(sector: int) -> List[int]:
    """Query MAST to get all TIC IDs observed in a given TESS sector.

    Uses ``astroquery.mast.Observations`` to find all 2-minute cadence
    timeseries observations for the specified sector.

    Parameters
    ----------
    sector : int
        TESS sector number.

    Returns
    -------
    list of int
        TIC IDs of all targets observed in the sector.
    """
    from astroquery.mast import Observations

    logger.info(f"Querying MAST for all targets in TESS Sector {sector}...")

    try:
        obs = Observations.query_criteria(
            project="TESS",
            obs_collection="TESS",
            dataproduct_type="timeseries",
            sequence_number=sector,
            t_exptime=[0, 200],  # 2-minute cadence
        )

        if obs is None or len(obs) == 0:
            logger.warning(f"MAST returned no observations for Sector {sector}.")
            return []

        # target_name column contains the TIC ID as a string
        tic_ids = []
        for name in obs["target_name"]:
            try:
                tic_ids.append(int(str(name).strip()))
            except (ValueError, TypeError):
                continue

        tic_ids = list(set(tic_ids))  # deduplicate
        logger.info(f"MAST returned {len(tic_ids)} unique targets for Sector {sector}.")
        return tic_ids

    except Exception as exc:
        logger.error(f"MAST query failed: {exc}")
        return []



@timer
def download_tess_lightcurves(
    sector: int = None,
    cadence: str = None,
    author: str = None,
    sample_size: Optional[int] = None,
    tic_list: List[int] = None,
    output_dir: Path = None,
) -> List[Path]:
    """Download PDCSAP flux light curves for a list of TESS targets.

    Parameters
    ----------
    sector : int, optional
        TESS sector number. Defaults to ``config.TESS_SECTOR``.
    cadence : str, optional
        Cadence type (``"short"`` for 2-min). Defaults to ``config.TESS_CADENCE``.
    author : str, optional
        Pipeline author (e.g. ``"SPOC"``). Defaults to ``config.TESS_AUTHOR``.
    sample_size : int or None, optional
        Maximum number of successful downloads to collect.
    tic_list : list of int, optional
        Specific TIC IDs to download (assumed to already exist in the sector).
    output_dir : Path, optional
        Directory to save raw FITS files. Defaults to ``config.RAW_DATA_DIR``.

    Returns
    -------
    list of Path
        Paths to the saved light-curve FITS files.
    """
    import lightkurve as lk

    sector = sector or config.TESS_SECTOR
    cadence = cadence or config.TESS_CADENCE
    author = author or config.TESS_AUTHOR
    output_dir = Path(output_dir or config.RAW_DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not tic_list:
        logger.warning("No TIC list provided. Falling back to a known test target (TIC 261136679 - Pi Mensae).")
        tic_list = [261136679]

    target_count = sample_size if sample_size is not None else len(tic_list)
    logger.info(
        f"Downloading up to {target_count} light curves from TESS Sector {sector} "
        f"({len(tic_list)} candidates available)"
    )

    saved_paths: List[Path] = []
    skipped = 0
    from src.utils import ProgressTracker
    tracker = ProgressTracker(target_count, description="Downloading LCs")

    for tic in tic_list:
        if sample_size is not None and len(saved_paths) >= sample_size:
            logger.info(f"Reached requested sample size of {sample_size} successful downloads.")
            break

        target_name = f"TIC {tic}"
        out_path = output_dir / f"TIC_{tic}_s{sector:04d}.fits"

        # Skip if already downloaded
        if out_path.exists():
            saved_paths.append(out_path)
            tracker.update()
            continue

        success = False
        for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
            try:
                search_result = lk.search_lightcurve(
                    target_name,
                    mission="TESS",
                    sector=sector,
                    cadence=cadence,
                    author=author,
                )

                if len(search_result) == 0:
                    break  # Not found, skip to next target

                lc = search_result[0].download()
                if lc is None:
                    raise RuntimeError("lightkurve returned None")

                lc.to_fits(out_path, overwrite=True)
                saved_paths.append(out_path)
                success = True
                break

            except Exception as exc:
                wait = RETRY_BACKOFF_SECONDS * attempt
                logger.debug(
                    f"Attempt {attempt}/{MAX_DOWNLOAD_RETRIES} failed for "
                    f"{target_name}: {exc}. Retrying in {wait:.0f}s..."
                )
                time.sleep(wait)

        if success:
            tracker.update()
        else:
            skipped += 1

    logger.info(
        f"Downloaded {len(saved_paths)} light curves (skipped {skipped}) -> {output_dir}"
    )
    return saved_paths



@timer
def download_toi_catalog(
    output_path: Path = None,
    url: str = TOI_CATALOG_URL,
) -> pd.DataFrame:
    """Download the TESS Objects of Interest (TOI) catalog from the
    NASA Exoplanet Archive.

    Parameters
    ----------
    output_path : Path, optional
        CSV file to save. Defaults to ``config.CATALOG_DIR / "toi_catalog.csv"``.
    url : str, optional
        TAP sync query URL.

    Returns
    -------
    pd.DataFrame
        TOI catalog with key columns retained.
    """
    output_path = Path(output_path or config.CATALOG_DIR / "toi_catalog.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading TOI catalog from NASA Exoplanet Archive...")

    for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
        try:
            df = pd.read_csv(url, comment="#")
            break
        except Exception as exc:
            wait = RETRY_BACKOFF_SECONDS * attempt
            logger.warning(
                f"TOI download attempt {attempt}/{MAX_DOWNLOAD_RETRIES} failed: "
                f"{exc}. Retrying in {wait:.0f}s..."
            )
            time.sleep(wait)
    else:
        logger.error("Failed to download TOI catalog after all retries.")
        return pd.DataFrame(columns=TOI_KEY_COLUMNS)

    # Keep only key columns (if they exist)
    available = [c for c in TOI_KEY_COLUMNS if c in df.columns]
    df_key = df[available].copy()

    # Ensure TIC ID is integer
    if "tid" in df_key.columns:
        df_key["tid"] = pd.to_numeric(df_key["tid"], errors="coerce").astype("Int64")

    df_key.to_csv(output_path, index=False)
    logger.info(
        f"TOI catalog saved: {len(df_key)} entries -> {output_path}"
    )
    return df_key



@timer
def download_eb_catalog(
    output_path: Path = None,
) -> pd.DataFrame:
    """Download an eclipsing-binary catalog.

    Strategy (in order):
      1. Query the TESS-EBs catalog from Vizier (catalog ``J/ApJS/258/16``).
      2. If Vizier fails, try the Villanova TESS-EB CSV endpoint.
      3. As a last resort, create a minimal synthetic EB list.

    Parameters
    ----------
    output_path : Path, optional
        CSV file to save. Defaults to ``config.CATALOG_DIR / "eb_catalog.csv"``.

    Returns
    -------
    pd.DataFrame
        Eclipsing-binary catalog with at least a ``TIC`` column.
    """
    output_path = Path(output_path or config.CATALOG_DIR / "eb_catalog.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_eb: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Strategy 1: Vizier (astroquery)
    # ------------------------------------------------------------------
    df_eb = _try_vizier_eb_catalog()

    # ------------------------------------------------------------------
    # Strategy 2: Villanova TESS-EB web CSV
    # ------------------------------------------------------------------
    if df_eb is None:
        df_eb = _try_villanova_eb_catalog()

    # ------------------------------------------------------------------
    # Strategy 3: Synthetic fallback
    # ------------------------------------------------------------------
    if df_eb is None:
        logger.warning("All EB catalog sources failed — generating synthetic EB list.")
        df_eb = _generate_synthetic_eb_list()

    # Normalise column name to "TIC"
    for col_candidate in ("TIC", "tic", "TIC_ID", "tic_id", "ID", "target"):
        if col_candidate in df_eb.columns:
            df_eb = df_eb.rename(columns={col_candidate: "TIC"})
            break

    if "TIC" in df_eb.columns:
        df_eb["TIC"] = pd.to_numeric(df_eb["TIC"], errors="coerce").astype("Int64")

    df_eb.to_csv(output_path, index=False)
    logger.info(f"EB catalog saved: {len(df_eb)} entries -> {output_path}")
    return df_eb


def _try_vizier_eb_catalog() -> Optional[pd.DataFrame]:
    """Attempt to fetch the TESS EB catalog from Vizier."""
    try:
        from astroquery.vizier import Vizier

        logger.info("Querying Vizier for TESS-EBs catalog (J/ApJS/258/16)...")
        vizier = Vizier(columns=["*"], row_limit=-1)
        tables = vizier.get_catalogs("J/ApJS/258/16")

        if tables and len(tables) > 0:
            df = tables[0].to_pandas()
            logger.info(f"Vizier returned {len(df)} EB entries.")
            return df
    except Exception as exc:
        logger.warning(f"Vizier EB query failed: {exc}")
    return None


def _try_villanova_eb_catalog() -> Optional[pd.DataFrame]:
    """Attempt to fetch EB data from the Villanova TESS-EB page."""
    villanova_url = (
        "http://tessebs.villanova.edu/static/catalog/eb_catalog.csv"
    )
    try:
        logger.info("Trying Villanova TESS-EB CSV endpoint...")
        df = pd.read_csv(villanova_url)
        logger.info(f"Villanova returned {len(df)} EB entries.")
        return df
    except Exception as exc:
        logger.warning(f"Villanova download failed: {exc}")
    return None


def _generate_synthetic_eb_list(n_ebs: int = 200) -> pd.DataFrame:
    """Create a small synthetic EB catalog for pipeline testing.

    Generates random TIC IDs and orbital periods representative of
    short-period eclipsing binaries discovered by TESS.

    Parameters
    ----------
    n_ebs : int
        Number of synthetic entries to generate.

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.default_rng(42)
    tic_ids = rng.integers(1_000_000, 500_000_000, size=n_ebs)
    periods = rng.uniform(0.2, 10.0, size=n_ebs)
    morphologies = rng.choice(["D", "SD", "OC"], size=n_ebs, p=[0.5, 0.3, 0.2])

    df = pd.DataFrame({
        "TIC": tic_ids,
        "period": np.round(periods, 6),
        "morphology": morphologies,
        "source": "synthetic",
    })
    logger.info(f"Generated {n_ebs} synthetic EB entries.")
    return df



@timer
def create_labeled_dataset(
    tic_ids: List[int],
    toi_catalog: pd.DataFrame,
    eb_catalog: pd.DataFrame,
    output_path: Path = None,
) -> pd.DataFrame:
    """Cross-match TIC IDs against the TOI and EB catalogs to produce labels.

    Label mapping:
        * Confirmed planets (CP) and planet candidates (PC) -> **PLANET**
        * False positives (FP) and false alarms (FA) -> **BLEND**
        * Eclipsing binaries -> **ECLIPSING_BINARY**
        * Everything else -> **OTHER**

    Parameters
    ----------
    tic_ids : list of int
        TIC identifiers to label.
    toi_catalog : pd.DataFrame
        TOI catalog (must contain ``tid`` and ``tfopwg_disp`` columns).
    eb_catalog : pd.DataFrame
        Eclipsing-binary catalog (must contain ``TIC`` column).
    output_path : Path, optional
        CSV file to write. Defaults to
        ``config.CATALOG_DIR / "labeled_dataset.csv"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``tic_id``, ``label``, ``source``, and any extra catalog
        metadata joined from TOI.
    """
    output_path = Path(output_path or config.CATALOG_DIR / "labeled_dataset.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build fast lookup sets
    toi_tics: Dict[int, Dict[str, Any]] = {}
    if "tid" in toi_catalog.columns:
        for _, row in toi_catalog.iterrows():
            tid = row["tid"]
            if pd.notna(tid):
                toi_tics[int(tid)] = row.to_dict()

    eb_tics: set = set()
    if "TIC" in eb_catalog.columns:
        eb_tics = set(
            eb_catalog["TIC"].dropna().astype(int).tolist()
        )

    records: List[Dict[str, Any]] = []

    for tic_id in tic_ids:
        if tic_id in toi_tics:
            info = toi_tics[tic_id]
            disp = str(info.get("tfopwg_disp", "")).strip().upper()

            if disp in ("CP", "KP", "PC"):
                label = "PLANET"
            elif disp in ("FP", "FA"):
                label = "BLEND"
            else:
                label = "OTHER"

            records.append({
                "tic_id": tic_id,
                "label": label,
                "source": "TOI",
                "toi_disp": disp,
                "pl_orbper": info.get("pl_orbper"),
                "pl_trandep": info.get("pl_trandep"),
                "pl_trandur": info.get("pl_trandur"),
                "pl_rade": info.get("pl_rade"),
            })

        elif tic_id in eb_tics:
            records.append({
                "tic_id": tic_id,
                "label": "ECLIPSING_BINARY",
                "source": "EB_CATALOG",
            })

        else:
            records.append({
                "tic_id": tic_id,
                "label": "OTHER",
                "source": "NONE",
            })

    df_labels = pd.DataFrame(records)

    # Summary statistics
    label_counts = df_labels["label"].value_counts()
    logger.info("Label distribution:")
    for lbl, cnt in label_counts.items():
        logger.info(f"  {lbl:20s}: {cnt:>6d}")

    df_labels.to_csv(output_path, index=False)
    logger.info(f"Labeled dataset saved: {len(df_labels)} entries -> {output_path}")
    return df_labels



@timer
def generate_synthetic_transits(
    n_samples: int = 500,
    n_points: int = 1000,
    noise_level: float = 0.001,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    """Generate synthetic planet-transit light curves using ``batman``.

    Each light curve is a normalised flux time series with a single transit
    event embedded in Gaussian noise.

    Parameters
    ----------
    n_samples : int
        Number of synthetic transits to generate.
    n_points : int
        Number of data points per light curve.
    noise_level : float
        Standard deviation of additive Gaussian noise (relative to
        normalised flux = 1).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    times : np.ndarray, shape ``(n_samples, n_points)``
    fluxes : np.ndarray, shape ``(n_samples, n_points)``
    params : list of dict
        Physical parameters for each synthetic transit.
    """
    import batman

    rng = np.random.default_rng(seed)
    times_all = np.empty((n_samples, n_points))
    fluxes_all = np.empty((n_samples, n_points))
    params_list: List[Dict[str, float]] = []

    logger.info(f"Generating {n_samples} synthetic planet transits...")

    for i in range(n_samples):
        # Randomise transit parameters
        period = rng.uniform(1.0, 15.0)                   # days
        rp_rs = rng.uniform(0.01, 0.15)                   # planet-to-star radius ratio
        a_rs = rng.uniform(3.0, 30.0)                     # semi-major axis / R_star
        inc = rng.uniform(85.0, 90.0)                     # inclination (degrees)
        ecc = 0.0                                         # circular orbit
        omega = 90.0                                      # argument of periastron
        t0 = rng.uniform(0.3, 0.7) * period               # mid-transit time
        u1, u2 = 0.3, 0.2                                 # limb darkening coeffs

        # Time array covering one full orbital period
        t = np.linspace(0.0, period, n_points)

        # batman model
        bm_params = batman.TransitParams()
        bm_params.t0 = t0
        bm_params.per = period
        bm_params.rp = rp_rs
        bm_params.a = a_rs
        bm_params.inc = inc
        bm_params.ecc = ecc
        bm_params.w = omega
        bm_params.limb_dark = config.LIMB_DARKENING_MODEL
        bm_params.u = [u1, u2]

        model = batman.TransitModel(bm_params, t)
        flux = model.light_curve(bm_params)

        # Add realistic Gaussian noise
        flux += rng.normal(0.0, noise_level, size=n_points)

        times_all[i] = t
        fluxes_all[i] = flux
        params_list.append({
            "period": period,
            "rp_rs": rp_rs,
            "a_rs": a_rs,
            "inc": inc,
            "t0": t0,
            "depth": rp_rs ** 2,
        })

    logger.info(f"Generated {n_samples} synthetic planet transits.")
    return times_all, fluxes_all, params_list


@timer
def generate_synthetic_ebs(
    n_samples: int = 300,
    n_points: int = 1000,
    noise_level: float = 0.001,
    seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    """Generate synthetic eclipsing-binary light curves.

    EB eclipses are modelled as deeper, V-shaped dips compared to planet
    transits.  A primary eclipse is always generated; a secondary eclipse
    is added at phase 0.5 with a shallower depth.

    Parameters
    ----------
    n_samples : int
        Number of synthetic EB light curves.
    n_points : int
        Number of data points per light curve.
    noise_level : float
        Standard deviation of additive Gaussian noise.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    times : np.ndarray, shape ``(n_samples, n_points)``
    fluxes : np.ndarray, shape ``(n_samples, n_points)``
    params : list of dict
        Parameters for each synthetic EB.
    """
    rng = np.random.default_rng(seed)
    times_all = np.empty((n_samples, n_points))
    fluxes_all = np.empty((n_samples, n_points))
    params_list: List[Dict[str, float]] = []

    logger.info(f"Generating {n_samples} synthetic eclipsing-binary light curves...")

    for i in range(n_samples):
        period = rng.uniform(0.3, 10.0)
        primary_depth = rng.uniform(0.05, 0.50)          # deep, EB-like
        secondary_depth = primary_depth * rng.uniform(0.0, 0.6)
        primary_width = rng.uniform(0.02, 0.10)          # phase width
        secondary_width = primary_width * rng.uniform(0.5, 1.5)
        t0 = rng.uniform(0.2, 0.8) * period

        t = np.linspace(0.0, period, n_points)
        phase = ((t - t0) / period) % 1.0

        # Start with flat baseline
        flux = np.ones(n_points)

        # Primary eclipse — V-shaped (linear ingress/egress)
        flux -= _v_shaped_eclipse(phase, center=0.0, depth=primary_depth,
                                  width=primary_width)

        # Secondary eclipse at phase 0.5
        if secondary_depth > 0.005:
            flux -= _v_shaped_eclipse(phase, center=0.5, depth=secondary_depth,
                                      width=secondary_width)

        # Add noise
        flux += rng.normal(0.0, noise_level, size=n_points)

        times_all[i] = t
        fluxes_all[i] = flux
        params_list.append({
            "period": period,
            "primary_depth": primary_depth,
            "secondary_depth": secondary_depth,
            "primary_width": primary_width,
            "t0": t0,
        })

    logger.info(f"Generated {n_samples} synthetic EB light curves.")
    return times_all, fluxes_all, params_list


def _v_shaped_eclipse(
    phase: np.ndarray,
    center: float,
    depth: float,
    width: float,
) -> np.ndarray:
    """Create a V-shaped (triangular) eclipse profile.

    Parameters
    ----------
    phase : np.ndarray
        Orbital phase array in [0, 1).
    center : float
        Phase of eclipse center.
    depth : float
        Maximum depth of the eclipse.
    width : float
        Full-width of the eclipse in phase units.

    Returns
    -------
    np.ndarray
        Eclipse flux decrement (positive values = dimming).
    """
    # Wrap phases relative to eclipse center into [-0.5, 0.5)
    delta = phase - center
    delta = delta - np.round(delta)
    half_w = width / 2.0
    eclipse = np.where(
        np.abs(delta) < half_w,
        depth * (1.0 - np.abs(delta) / half_w),
        0.0,
    )
    return eclipse



@timer
def run_full_acquisition(
    sector: int = None,
    sample_size: Optional[int] = None,
    skip_lightcurves: bool = False,
    n_synthetic_planets: int = 500,
    n_synthetic_ebs: int = 300,
) -> Dict[str, Any]:
    """Run the complete data-acquisition pipeline end to end.

    Steps executed:
      1. Download TOI catalog.
      2. Download EB catalog.
      3. (Optional) Download TESS light curves for the sector.
      4. Create a labeled dataset by cross-matching TIC IDs.
      5. Generate synthetic training data (planets + EBs).

    Parameters
    ----------
    sector : int, optional
        TESS sector. Defaults to ``config.TESS_SECTOR``.
    sample_size : int or None, optional
        Limit on light-curve downloads. Defaults to ``config.SAMPLE_SIZE``.
    skip_lightcurves : bool
        If ``True``, skip the (slow) light-curve download step.
    n_synthetic_planets : int
        Number of synthetic planet transits to generate.
    n_synthetic_ebs : int
        Number of synthetic EB light curves to generate.

    Returns
    -------
    dict
        Summary containing paths and counts for each artefact produced.
    """
    sector = sector or config.TESS_SECTOR
    sample_size = sample_size if sample_size is not None else config.SAMPLE_SIZE

    results: Dict[str, Any] = {"sector": sector}

    # --- Step 1: TOI catalog ---
    toi_catalog = download_toi_catalog()
    results["toi_catalog_entries"] = len(toi_catalog)

    # --- Step 2: EB catalog ---
    eb_catalog = download_eb_catalog()
    results["eb_catalog_entries"] = len(eb_catalog)

    # --- Step 3: Light curves ---
    tic_ids: List[int] = []
    
    # Extract TIC IDs from the TOI catalog to search for actual planet candidates
    if "tid" in toi_catalog.columns:
        tic_ids = toi_catalog["tid"].dropna().astype(int).unique().tolist()
        
    if not skip_lightcurves:
        saved_files = download_tess_lightcurves(
            sector=sector, sample_size=sample_size, tic_list=tic_ids
        )
        results["lightcurves_downloaded"] = len(saved_files)

        # Update tic_ids list to only contain what we actually downloaded
        from src.utils import tic_id_from_filename
        tic_ids = [
            tic_id_from_filename(p.name)
            for p in saved_files
            if tic_id_from_filename(p.name) is not None
        ]
    else:
        logger.info("Skipping light-curve download (skip_lightcurves=True).")
        # Use TIC IDs from the TOI catalog for labelling demo
        if "tid" in toi_catalog.columns:
            tic_ids = toi_catalog["tid"].dropna().astype(int).tolist()[:100]

    results["tic_ids_count"] = len(tic_ids)

    # --- Step 4: Labeled dataset ---
    if tic_ids:
        labeled_df = create_labeled_dataset(tic_ids, toi_catalog, eb_catalog)
        results["labeled_dataset_entries"] = len(labeled_df)
    else:
        logger.warning("No TIC IDs available — skipping labeled dataset creation.")
        results["labeled_dataset_entries"] = 0

    # --- Step 5: Synthetic data ---
    syn_planet_times, syn_planet_fluxes, syn_planet_params = (
        generate_synthetic_transits(n_samples=n_synthetic_planets)
    )
    save_pickle(
        {"times": syn_planet_times, "fluxes": syn_planet_fluxes,
         "params": syn_planet_params},
        config.PROCESSED_DATA_DIR / "synthetic_planets.pkl",
    )
    results["synthetic_planets"] = n_synthetic_planets

    syn_eb_times, syn_eb_fluxes, syn_eb_params = (
        generate_synthetic_ebs(n_samples=n_synthetic_ebs)
    )
    save_pickle(
        {"times": syn_eb_times, "fluxes": syn_eb_fluxes,
         "params": syn_eb_params},
        config.PROCESSED_DATA_DIR / "synthetic_ebs.pkl",
    )
    results["synthetic_ebs"] = n_synthetic_ebs

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("DATA ACQUISITION COMPLETE")
    logger.info("=" * 60)
    for key, val in results.items():
        logger.info(f"  {key:30s}: {val}")

    return results



if __name__ == "__main__":
    """Run a small sample acquisition for quick testing / development."""

    logger.info("Running data_acquisition in standalone mode (small sample)...")

    summary = run_full_acquisition(
        sector=config.TESS_SECTOR,
        sample_size=10,             # only 10 light curves for a quick test
        skip_lightcurves=False,
        n_synthetic_planets=100,
        n_synthetic_ebs=50,
    )

    logger.info("Standalone run finished.")
    logger.info(f"Summary: {summary}")
