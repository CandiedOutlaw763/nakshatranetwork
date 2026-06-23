"""
Exoplanet Detection Pipeline — AI-driven analysis of TESS light curves.

Modules:
    data_acquisition    — Download TESS data and catalogs from MAST
    preprocessing       — Clean, detrend, and normalize light curves
    signal_detection    — BLS/TLS periodogram transit search
    feature_extraction  — Phase folding and diagnostic feature computation
    classifier          — CNN + Random Forest classification models
    transit_fitting     — Physical transit model fitting with batman
    snr_calculator      — Signal-to-noise and confidence metrics
    visualization       — Publication-quality plotting functions
    utils               — Shared helper functions
"""

__version__ = "1.0.0"
__author__ = "Exoplanet Detection Team"
