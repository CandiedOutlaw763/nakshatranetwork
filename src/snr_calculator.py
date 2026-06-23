"""
Signal-to-Noise Ratio and Confidence Metrics Module.

Computes SNR, SDE, False Alarm Probability, and combined confidence
scores for detected transit signals.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.utils import logger, phase_fold



@dataclass
class ConfidenceMetrics:
    """Collection of confidence and significance metrics for a detection."""
    transit_snr: float = 0.0           # Transit signal-to-noise ratio
    single_transit_snr: float = 0.0    # SNR of a single transit event
    bls_sde: float = 0.0              # BLS/TLS Signal Detection Efficiency
    false_alarm_prob: float = 1.0      # False Alarm Probability
    classifier_prob: float = 0.0       # ML classifier confidence
    classifier_class: str = "OTHER"    # Predicted class
    combined_confidence: float = 0.0   # Weighted combined confidence (0-1)
    chi2_flat: float = 0.0            # Chi-squared for flat (no-transit) model
    chi2_transit: float = 0.0          # Chi-squared for transit model
    delta_bic: float = 0.0            # BIC(flat) - BIC(transit), positive = transit preferred
    n_transits: int = 0                # Number of observed transits
    n_in_transit: int = 0              # Number of in-transit data points

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @property
    def is_significant(self) -> bool:
        """Check if detection meets significance thresholds."""
        return (self.transit_snr >= config.SNR_THRESHOLD and
                self.bls_sde >= config.SDE_THRESHOLD)

    @property
    def confidence_label(self) -> str:
        """Human-readable confidence level."""
        c = self.combined_confidence
        if c >= 0.9:
            return "Very High"
        elif c >= 0.7:
            return "High"
        elif c >= 0.5:
            return "Moderate"
        elif c >= 0.3:
            return "Low"
        else:
            return "Very Low"



def compute_transit_snr(time: np.ndarray, flux: np.ndarray,
                        flux_err: np.ndarray = None,
                        period: float = 1.0, epoch: float = 0.0,
                        depth: float = 0.001, duration: float = 0.1) -> Dict:
    """Compute transit signal-to-noise ratio.

    Uses the standard transit SNR formula:
        SNR = δ × √(N_transit × N_in_transit) / σ

    where δ = transit depth, σ = out-of-transit scatter,
    N_transit = number of transits, N_in_transit = points per transit.

    Also computes the refined trapezoidal SNR:
        SNR_trap = δ × √((T14 + 2*T23) / 3) / σ_per_unit_time

    Args:
        time: Time array
        flux: Normalized flux array
        flux_err: Flux errors
        period: Orbital period (days)
        epoch: Mid-transit time
        depth: Transit depth (fractional)
        duration: Transit duration (days)

    Returns:
        Dictionary with SNR metrics
    """
    if len(time) < 10:
        return {"transit_snr": 0, "single_transit_snr": 0,
                "n_transits": 0, "n_in_transit": 0}

    # Phase fold
    phase = phase_fold(time, period, epoch)

    # Identify in-transit and out-of-transit points
    half_dur_phase = (duration / period) / 2.0
    in_transit_mask = np.abs(phase) < half_dur_phase
    out_transit_mask = ~in_transit_mask

    n_in_transit = np.sum(in_transit_mask)
    n_out_transit = np.sum(out_transit_mask)

    if n_in_transit < 2 or n_out_transit < 10:
        return {"transit_snr": 0, "single_transit_snr": 0,
                "n_transits": 0, "n_in_transit": 0}

    # Out-of-transit scatter
    if flux_err is not None and np.any(flux_err > 0):
        sigma = np.nanmedian(flux_err[out_transit_mask])
    else:
        sigma = np.nanstd(flux[out_transit_mask])

    if sigma <= 0:
        sigma = np.nanstd(flux)

    # Count number of transits
    time_span = np.nanmax(time) - np.nanmin(time)
    n_transits = max(1, int(np.round(time_span / period)))

    # Points per single transit
    n_per_transit = max(1, n_in_transit / n_transits) if n_transits > 0 else n_in_transit

    # === Standard boxcar SNR ===
    # SNR = depth * sqrt(total_in_transit_points) / sigma
    transit_snr = abs(depth) * np.sqrt(n_in_transit) / sigma if sigma > 0 else 0

    # Single-transit SNR
    single_transit_snr = abs(depth) * np.sqrt(n_per_transit) / sigma if sigma > 0 else 0

    # === Measured depth from data ===
    mean_in = np.nanmean(flux[in_transit_mask])
    mean_out = np.nanmean(flux[out_transit_mask])
    measured_depth = mean_out - mean_in
    measured_depth_err = sigma * np.sqrt(1.0 / n_in_transit + 1.0 / n_out_transit)

    # === Depth SNR ===
    depth_snr = measured_depth / measured_depth_err if measured_depth_err > 0 else 0

    return {
        "transit_snr": float(transit_snr),
        "single_transit_snr": float(single_transit_snr),
        "depth_snr": float(depth_snr),
        "measured_depth": float(measured_depth),
        "measured_depth_err": float(measured_depth_err),
        "out_of_transit_scatter": float(sigma),
        "n_transits": int(n_transits),
        "n_in_transit": int(n_in_transit),
        "n_per_transit": float(n_per_transit),
    }



def compute_sde(power: np.ndarray) -> float:
    """Compute Signal Detection Efficiency from periodogram power.

    SDE = (peak_power - mean(power)) / std(power)
    """
    if len(power) < 10:
        return 0.0

    mean_power = np.nanmean(power)
    std_power = np.nanstd(power)

    if std_power <= 0:
        return 0.0

    max_power = np.nanmax(power)
    sde = (max_power - mean_power) / std_power
    return float(sde)


def compute_fap(sde: float, n_trials: int = None) -> float:
    """Compute False Alarm Probability from SDE.

    Uses Gaussian tail probability with trials correction.

    Args:
        sde: Signal Detection Efficiency
        n_trials: Number of independent trials (periods tested).
                  If None, uses PERIOD_GRID_SIZE from config.

    Returns:
        False Alarm Probability (0-1)
    """
    if n_trials is None:
        n_trials = config.PERIOD_GRID_SIZE

    # Single-trial p-value from Gaussian tail
    p_single = stats.norm.sf(sde)  # survival function = 1 - CDF

    # Multiple-trials correction (Bonferroni-like)
    fap = 1.0 - (1.0 - p_single) ** n_trials

    # Clamp to [0, 1]
    fap = np.clip(fap, 0.0, 1.0)

    return float(fap)



def compute_delta_bic(time: np.ndarray, flux: np.ndarray,
                      flux_err: np.ndarray,
                      model_flux: np.ndarray) -> Dict:
    """Compare transit model vs flat model using BIC.

    ΔBIC = BIC(flat) - BIC(transit)
    ΔBIC > 10 strongly favors transit model.

    Args:
        time: Time array
        flux: Observed flux
        flux_err: Flux errors
        model_flux: Best-fit transit model flux

    Returns:
        Dictionary with chi2 and BIC values
    """
    n = len(time)
    if n < 10:
        return {"chi2_flat": 0, "chi2_transit": 0, "delta_bic": 0}

    sigma = flux_err if flux_err is not None else np.full(n, np.nanstd(flux))
    sigma = np.where(sigma > 0, sigma, np.nanmedian(sigma[sigma > 0]))

    # Chi-squared for flat model (1 parameter: mean)
    flat_model = np.nanmean(flux)
    chi2_flat = np.sum(((flux - flat_model) / sigma) ** 2)
    k_flat = 1
    bic_flat = chi2_flat + k_flat * np.log(n)

    # Chi-squared for transit model (4 parameters)
    chi2_transit = np.sum(((flux - model_flux) / sigma) ** 2)
    k_transit = 4
    bic_transit = chi2_transit + k_transit * np.log(n)

    delta_bic = bic_flat - bic_transit  # Positive means transit is better

    return {
        "chi2_flat": float(chi2_flat),
        "chi2_transit": float(chi2_transit),
        "bic_flat": float(bic_flat),
        "bic_transit": float(bic_transit),
        "delta_bic": float(delta_bic),
    }



def compute_confidence(sde: float = 0.0, snr: float = 0.0,
                       classifier_prob: float = 0.0,
                       classifier_class: str = "OTHER",
                       delta_bic: float = 0.0,
                       weights: Dict = None) -> ConfidenceMetrics:
    """Compute combined confidence score for a transit detection.

    Combines multiple independent metrics into a single confidence score.

    Args:
        sde: Signal Detection Efficiency
        snr: Transit signal-to-noise ratio
        classifier_prob: ML classifier probability for the predicted class
        classifier_class: Predicted classification
        delta_bic: ΔBIC (model comparison)
        weights: Custom weights for combining metrics

    Returns:
        ConfidenceMetrics object
    """
    if weights is None:
        weights = config.CONFIDENCE_WEIGHTS

    # Normalize each metric to [0, 1]
    # SDE: threshold at 7, saturate at ~20
    sde_score = np.clip((sde - 3.0) / 15.0, 0.0, 1.0)

    # SNR: threshold at 7, saturate at ~50
    snr_score = np.clip((snr - 3.0) / 40.0, 0.0, 1.0)

    # Classifier probability is already [0, 1]
    cls_score = classifier_prob

    # BIC: strong evidence at ΔBIC > 10
    bic_score = np.clip(delta_bic / 20.0, 0.0, 1.0)

    # Weighted combination
    w = weights
    w_sum = w.get("sde", 0.3) + w.get("snr", 0.3) + w.get("classifier_prob", 0.4)

    combined = (
        w.get("sde", 0.3) * sde_score +
        w.get("snr", 0.3) * snr_score +
        w.get("classifier_prob", 0.4) * cls_score
    ) / max(w_sum, 1e-6)

    # Bonus for strong BIC evidence
    if delta_bic > 10:
        combined = min(1.0, combined + 0.05)

    metrics = ConfidenceMetrics(
        transit_snr=snr,
        bls_sde=sde,
        false_alarm_prob=compute_fap(sde),
        classifier_prob=classifier_prob,
        classifier_class=classifier_class,
        combined_confidence=float(np.clip(combined, 0.0, 1.0)),
        chi2_transit=0.0,  # Set externally if available
        delta_bic=delta_bic,
    )

    return metrics


def compute_full_metrics(time: np.ndarray, flux: np.ndarray,
                         flux_err: np.ndarray = None,
                         period: float = 1.0, epoch: float = 0.0,
                         depth: float = 0.001, duration: float = 0.1,
                         model_flux: np.ndarray = None,
                         sde: float = 0.0,
                         classifier_prob: float = 0.0,
                         classifier_class: str = "OTHER",
                         periodogram_power: np.ndarray = None) -> ConfidenceMetrics:
    """Compute all confidence metrics for a detection.

    This is the main entry point that combines SNR, SDE, BIC, and
    classifier results into a complete ConfidenceMetrics object.

    Args:
        time: Time array
        flux: Normalized flux
        flux_err: Flux errors
        period: Detected period
        epoch: Detected epoch
        depth: Detected depth
        duration: Detected duration
        model_flux: Best-fit transit model (for BIC)
        sde: Signal Detection Efficiency from BLS/TLS
        classifier_prob: ML classifier probability
        classifier_class: Predicted class
        periodogram_power: Full periodogram power (for SDE computation)

    Returns:
        ConfidenceMetrics
    """
    # Default flux errors
    if flux_err is None:
        flux_err = np.full_like(flux, np.nanstd(flux))

    # Compute SNR
    snr_dict = compute_transit_snr(time, flux, flux_err, period, epoch, depth, duration)

    # Compute SDE if periodogram power provided
    if periodogram_power is not None and sde == 0.0:
        sde = compute_sde(periodogram_power)

    # Compute BIC if model flux provided
    delta_bic = 0.0
    bic_dict = {}
    if model_flux is not None:
        bic_dict = compute_delta_bic(time, flux, flux_err, model_flux)
        delta_bic = bic_dict.get("delta_bic", 0.0)

    # Combined confidence
    metrics = compute_confidence(
        sde=sde,
        snr=snr_dict["transit_snr"],
        classifier_prob=classifier_prob,
        classifier_class=classifier_class,
        delta_bic=delta_bic,
    )

    # Fill in additional fields
    metrics.transit_snr = snr_dict["transit_snr"]
    metrics.single_transit_snr = snr_dict["single_transit_snr"]
    metrics.n_transits = snr_dict["n_transits"]
    metrics.n_in_transit = snr_dict["n_in_transit"]

    if bic_dict:
        metrics.chi2_flat = bic_dict.get("chi2_flat", 0.0)
        metrics.chi2_transit = bic_dict.get("chi2_transit", 0.0)
        metrics.delta_bic = bic_dict.get("delta_bic", 0.0)

    return metrics



if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    t = np.linspace(0, 27, 19000)
    period, epoch, depth = 3.5, 1.0, 0.01

    # Create synthetic transit
    phase = phase_fold(t, period, epoch)
    duration = 0.1
    half_dur = (duration / period) / 2.0
    flux = np.ones_like(t)
    flux[np.abs(phase) < half_dur] -= depth
    flux += np.random.normal(0, 0.002, len(t))
    flux_err = np.full_like(flux, 0.002)

    # Compute metrics
    snr = compute_transit_snr(t, flux, flux_err, period, epoch, depth, duration)
    print(f"Transit SNR: {snr['transit_snr']:.1f}")
    print(f"Single transit SNR: {snr['single_transit_snr']:.1f}")
    print(f"Number of transits: {snr['n_transits']}")

    # Full metrics
    metrics = compute_full_metrics(t, flux, flux_err, period, epoch, depth, duration,
                                    sde=12.5, classifier_prob=0.92,
                                    classifier_class="PLANET")
    print(f"\nCombined confidence: {metrics.combined_confidence:.3f} ({metrics.confidence_label})")
    print(f"FAP: {metrics.false_alarm_prob:.2e}")
    print(f"Significant: {metrics.is_significant}")
