"""
Global configuration for the Exoplanet Detection Pipeline.
All tunable parameters, paths, and constants are centralized here.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CATALOG_DIR = DATA_DIR / "catalogs"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
REPORT_DIR = PROJECT_ROOT / "report"

# Create directories
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CATALOG_DIR,
          MODELS_DIR, RESULTS_DIR, PLOTS_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TESS_SECTOR = 1                    # Default sector to download
TESS_CADENCE = "short"             # "short" = 2-min, "fast" = 20-sec
TESS_AUTHOR = "SPOC"              # Pipeline: SPOC for 2-min cadence
SAMPLE_SIZE = None                 # None = full sector; int = subset for testing
FLUX_COLUMN = "pdcsap_flux"        # Which flux column to use

SIGMA_CLIP_THRESHOLD = 5.0         # Sigma clipping for outlier removal
FLATTEN_WINDOW_LENGTH = 401        # Savitzky-Golay / biweight window (cadences)
DETREND_METHOD = "biweight"        # "biweight" (wotan) or "savgol"
QUALITY_BITMASK = "default"        # lightkurve quality bitmask

PERIOD_MIN = 0.5                   # Minimum search period (days)
PERIOD_MAX = 15.0                  # Maximum search period (days)
PERIOD_GRID_SIZE = 50000           # Number of trial periods
DURATION_MIN = 0.01                # Minimum transit duration (days)
DURATION_MAX = 0.2                 # Maximum transit duration (days)
N_DURATIONS = 20                   # Number of trial durations
SDE_THRESHOLD = 7.0                # Minimum SDE for a detection
MAX_PLANETS_PER_STAR = 3           # Max signals to search per light curve
USE_TLS = True                     # Use Transit Least Squares (better but slower)

GLOBAL_VIEW_BINS = 201             # Number of bins for global phase-folded view
LOCAL_VIEW_BINS = 61               # Number of bins for local (zoomed) transit view
LOCAL_VIEW_HALF_WIDTH = 2.0        # Local view width in transit durations

# Feature thresholds for diagnostics
SECONDARY_ECLIPSE_THRESHOLD = 0.001  # Min depth to flag secondary eclipse
ODD_EVEN_RATIO_THRESHOLD = 3.0      # Sigma threshold for odd/even depth diff

CLASSIFICATION_CLASSES = ["PLANET", "ECLIPSING_BINARY", "BLEND", "OTHER"]
N_CLASSES = len(CLASSIFICATION_CLASSES)

# CNN Architecture
CNN_LEARNING_RATE = 1e-3
CNN_BATCH_SIZE = 64
CNN_EPOCHS = 100
CNN_EARLY_STOPPING_PATIENCE = 10
CNN_DROPOUT_RATE = 0.3
CNN_GLOBAL_FILTERS = [16, 32, 64]   # Conv1D filter counts for global branch
CNN_LOCAL_FILTERS = [16, 32, 64]     # Conv1D filter counts for local branch
CNN_DENSE_UNITS = [128, 64]          # Dense layer units after concatenation

# Random Forest
RF_N_ESTIMATORS = 500
RF_MAX_DEPTH = 15
RF_MIN_SAMPLES_SPLIT = 5
RF_CLASS_WEIGHT = "balanced"

# Ensemble weights
CNN_WEIGHT = 0.7                     # Weight for CNN predictions in ensemble
RF_WEIGHT = 0.3                      # Weight for RF predictions in ensemble

# Train/Val/Test split
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

LIMB_DARKENING_MODEL = "quadratic"
DEFAULT_LIMB_DARKENING_COEFFS = [0.3, 0.2]  # Default u1, u2
FIT_METHOD = "Nelder-Mead"          # scipy.optimize method
MCMC_NWALKERS = 32                   # emcee walkers
MCMC_NSTEPS = 2000                   # emcee steps
MCMC_BURNIN = 500                    # burn-in steps to discard
USE_MCMC = False                     # Use MCMC for uncertainties (slower)
N_BOOTSTRAP = 100                    # Bootstrap iterations for quick uncertainties

SNR_THRESHOLD = 7.0                  # Minimum SNR for confident detection
CONFIDENCE_WEIGHTS = {
    "sde": 0.3,
    "snr": 0.3,
    "classifier_prob": 0.4,
}

DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8050
PLOT_STYLE = "dark_background"       # matplotlib style
PLOT_DPI = 150
FIGURE_SIZE = (14, 10)

# Color palette for the dashboard
COLORS = {
    "bg_primary": "#0a0e17",
    "bg_secondary": "#131a2a",
    "bg_card": "#1a2332",
    "accent_blue": "#00d4ff",
    "accent_purple": "#7c3aed",
    "accent_green": "#10b981",
    "accent_red": "#ef4444",
    "accent_yellow": "#f59e0b",
    "accent_orange": "#f97316",
    "text_primary": "#e2e8f0",
    "text_secondary": "#94a3b8",
    "grid": "#1e293b",
    "planet": "#10b981",
    "eb": "#ef4444",
    "blend": "#f59e0b",
    "other": "#6b7280",
}

LOG_LEVEL = "INFO"
LOG_FILE = RESULTS_DIR / "pipeline.log"
