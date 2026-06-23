"""
Transit Model Fitting Module.

Fits physical transit models to light curves using the batman package.
Estimates orbital/transit parameters with uncertainties via bootstrap or MCMC.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize, differential_evolution

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.utils import logger, timer, phase_fold, save_json

try:
    import batman
    HAS_BATMAN = True
except ImportError:
    HAS_BATMAN = False
    logger.warning("batman-package not installed. Transit fitting will use analytic approximation.")

try:
    import emcee
    HAS_EMCEE = True
except ImportError:
    HAS_EMCEE = False



@dataclass
class TransitParams:
    """Estimated transit parameters with uncertainties."""
    period: float = 0.0              # Orbital period (days)
    period_err: float = 0.0
    epoch: float = 0.0               # Mid-transit time (BJD)
    epoch_err: float = 0.0
    depth: float = 0.0               # Transit depth (fractional)
    depth_err: float = 0.0
    depth_ppm: float = 0.0           # Transit depth (parts per million)
    depth_ppm_err: float = 0.0
    duration: float = 0.0            # Transit duration (days)
    duration_err: float = 0.0
    duration_hours: float = 0.0      # Transit duration (hours)
    duration_hours_err: float = 0.0
    rp_rs: float = 0.0              # Planet-to-star radius ratio
    rp_rs_err: float = 0.0
    a_rs: float = 0.0               # Semi-major axis / stellar radius
    a_rs_err: float = 0.0
    inclination: float = 0.0         # Orbital inclination (degrees)
    inclination_err: float = 0.0
    impact_param: float = 0.0        # Impact parameter
    impact_param_err: float = 0.0
    u1: float = 0.3                  # Limb darkening coeff 1
    u2: float = 0.2                  # Limb darkening coeff 2
    chi_squared: float = 0.0         # Goodness of fit
    reduced_chi_squared: float = 0.0
    bic: float = 0.0                # Bayesian Information Criterion
    model_flux: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> Dict:
        """Convert to dictionary (excluding model flux array)."""
        d = {}
        for key, val in self.__dict__.items():
            if key != "model_flux" and not key.startswith("_"):
                d[key] = val
        return d

    def summary(self) -> str:
        """Human-readable parameter summary."""
        lines = [
            "═══════════════════════════════════════════",
            "       TRANSIT PARAMETER ESTIMATES         ",
            "═══════════════════════════════════════════",
            f"  Period:        {self.period:.6f} ± {self.period_err:.6f} days",
            f"  Epoch (T₀):    {self.epoch:.6f} ± {self.epoch_err:.6f} BJD",
            f"  Depth:         {self.depth_ppm:.1f} ± {self.depth_ppm_err:.1f} ppm",
            f"  Duration:      {self.duration_hours:.2f} ± {self.duration_hours_err:.2f} hours",
            f"  Rp/Rs:         {self.rp_rs:.4f} ± {self.rp_rs_err:.4f}",
            f"  a/Rs:          {self.a_rs:.2f} ± {self.a_rs_err:.2f}",
            f"  Inclination:   {self.inclination:.2f} ± {self.inclination_err:.2f}°",
            f"  Impact param:  {self.impact_param:.3f} ± {self.impact_param_err:.3f}",
            f"  χ²_red:        {self.reduced_chi_squared:.3f}",
            "═══════════════════════════════════════════",
        ]
        return "\n".join(lines)



def _batman_model(time: np.ndarray, period: float, epoch: float,
                  rp_rs: float, a_rs: float, inc: float,
                  u1: float = 0.3, u2: float = 0.2,
                  ecc: float = 0.0, omega: float = 90.0) -> np.ndarray:
    """Generate transit model flux using batman.

    Args:
        time: Time array
        period: Orbital period (days)
        epoch: Mid-transit time
        rp_rs: Planet-to-star radius ratio
        a_rs: Semi-major axis in stellar radii
        inc: Orbital inclination (degrees)
        u1, u2: Quadratic limb darkening coefficients
        ecc: Eccentricity
        omega: Argument of periastron (degrees)

    Returns:
        Model flux array
    """
    if not HAS_BATMAN:
        return _analytic_transit_model(time, period, epoch, rp_rs, a_rs, inc)

    params = batman.TransitParams()
    params.t0 = epoch
    params.per = period
    params.rp = rp_rs
    params.a = a_rs
    params.inc = inc
    params.ecc = ecc
    params.w = omega
    params.limb_dark = config.LIMB_DARKENING_MODEL
    params.u = [u1, u2]

    m = batman.TransitModel(params, time, transittype="primary")
    return m.light_curve(params)


def _analytic_transit_model(time: np.ndarray, period: float, epoch: float,
                            rp_rs: float, a_rs: float, inc: float) -> np.ndarray:
    """Simple analytic box transit model (fallback when batman unavailable)."""
    phase = phase_fold(time, period, epoch)
    depth = rp_rs ** 2

    # Approximate duration from a/Rs and inclination
    inc_rad = np.radians(inc)
    b = a_rs * np.cos(inc_rad)
    duration_phase = (1.0 / np.pi) * np.arcsin(
        np.sqrt((1 + rp_rs) ** 2 - b ** 2) / (a_rs * np.sin(inc_rad))
    ) if a_rs * np.sin(inc_rad) > 0 else 0.05

    flux = np.ones_like(time)
    in_transit = np.abs(phase) < duration_phase / 2.0
    flux[in_transit] = 1.0 - depth

    return flux



def _chi_squared(params_vec: np.ndarray, time: np.ndarray,
                 flux: np.ndarray, flux_err: np.ndarray,
                 period: float, fixed_period: bool = True) -> float:
    """Compute chi-squared for transit model fit.

    params_vec = [rp_rs, a_rs, inc, epoch] or [rp_rs, a_rs, inc, epoch, period]
    """
    rp_rs, a_rs, inc, epoch = params_vec[:4]
    p = period if fixed_period else params_vec[4]

    # Physical constraints
    if rp_rs <= 0 or rp_rs > 0.5:
        return 1e10
    if a_rs < 1.5 or a_rs > 200:
        return 1e10
    if inc < 60 or inc > 90:
        return 1e10

    try:
        model = _batman_model(time, p, epoch, rp_rs, a_rs, inc)
        residuals = (flux - model) / flux_err
        return np.sum(residuals ** 2)
    except Exception:
        return 1e10


def _initial_guess(depth: float, duration: float, period: float,
                   epoch: float) -> np.ndarray:
    """Generate initial parameter guess from BLS results.

    Args:
        depth: Transit depth (fractional)
        duration: Transit duration (days)
        period: Orbital period (days)
        epoch: Mid-transit time

    Returns:
        Array [rp_rs, a_rs, inc, epoch]
    """
    rp_rs = np.sqrt(max(depth, 1e-6))

    # Estimate a/Rs from duration and period
    # T ≈ (P/π) * arcsin(1/a_rs) for central transits
    duration_frac = duration / period
    if duration_frac > 0 and duration_frac < 0.5:
        a_rs_est = 1.0 / np.sin(np.pi * duration_frac)
        a_rs = np.clip(a_rs_est, 2.0, 100.0)
    else:
        a_rs = 10.0

    inc = 89.0  # Assume near-edge-on
    return np.array([rp_rs, a_rs, inc, epoch])


@timer
def fit_transit(time: np.ndarray, flux: np.ndarray,
                flux_err: np.ndarray = None,
                period: float = 1.0, epoch: float = 0.0,
                depth: float = 0.001, duration: float = 0.1,
                fit_period: bool = False) -> TransitParams:
    """Fit a transit model to a light curve.

    Uses scipy.optimize with batman model to find best-fit parameters,
    then estimates uncertainties via bootstrap resampling.

    Args:
        time: Time array (BJD)
        flux: Normalized flux array
        flux_err: Flux errors (estimated from scatter if None)
        period: Initial period estimate (days)
        epoch: Initial epoch estimate (BJD)
        depth: Initial depth estimate (fractional)
        duration: Initial duration estimate (days)
        fit_period: If True, also fit the period

    Returns:
        TransitParams with best-fit values and uncertainties
    """
    # Default errors from data scatter
    if flux_err is None:
        flux_err = np.full_like(flux, np.nanstd(flux) * 0.5)
    flux_err = np.where(flux_err > 0, flux_err, np.nanmedian(flux_err[flux_err > 0]))

    # Remove NaN
    mask = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    time, flux, flux_err = time[mask], flux[mask], flux_err[mask]

    if len(time) < 20:
        logger.warning("Too few data points for transit fitting")
        return _quick_estimate(period, epoch, depth, duration)

    # Initial guess
    x0 = _initial_guess(depth, duration, period, epoch)
    logger.debug(f"Initial guess: rp_rs={x0[0]:.4f}, a_rs={x0[1]:.1f}, inc={x0[2]:.1f}")

    # --- Optimize ---
    try:
        # Bounds for differential evolution
        bounds = [
            (0.001, 0.4),     # rp_rs
            (2.0, 100.0),     # a_rs
            (70.0, 90.0),     # inc
            (epoch - 0.1 * period, epoch + 0.1 * period),  # epoch
        ]

        # First pass: differential evolution (global search)
        result_de = differential_evolution(
            _chi_squared, bounds,
            args=(time, flux, flux_err, period, True),
            maxiter=200, tol=1e-6, seed=42,
            popsize=15, mutation=(0.5, 1.5)
        )

        # Second pass: Nelder-Mead refinement
        result = minimize(
            _chi_squared, result_de.x,
            args=(time, flux, flux_err, period, True),
            method=config.FIT_METHOD,
            options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8}
        )

        best_params = result.x
        best_chi2 = result.fun
    except Exception as e:
        logger.warning(f"Optimization failed: {e}. Using initial guess.")
        best_params = x0
        best_chi2 = _chi_squared(x0, time, flux, flux_err, period)

    rp_rs, a_rs, inc, epoch_fit = best_params

    # --- Generate best-fit model ---
    model_flux = _batman_model(time, period, epoch_fit, rp_rs, a_rs, inc)

    # --- Compute uncertainties via bootstrap ---
    param_samples = _bootstrap_uncertainties(
        time, flux, flux_err, period, best_params,
        n_bootstrap=config.N_BOOTSTRAP
    )

    # --- Build result ---
    result = TransitParams()
    result.period = period
    result.epoch = epoch_fit
    result.rp_rs = rp_rs
    result.a_rs = a_rs
    result.inclination = inc

    # Derived parameters
    result.depth = rp_rs ** 2
    result.depth_ppm = result.depth * 1e6

    # Impact parameter: b = a_rs * cos(inc)
    result.impact_param = a_rs * np.cos(np.radians(inc))

    # Duration estimate: T ≈ (P/π) * arcsin(sqrt((1+rp_rs)^2 - b^2) / (a_rs * sin(i)))
    b = result.impact_param
    sin_i = np.sin(np.radians(inc))
    arg = np.sqrt(max((1 + rp_rs) ** 2 - b ** 2, 0)) / (a_rs * sin_i) if a_rs * sin_i > 0 else 0
    if 0 < arg <= 1:
        result.duration = (period / np.pi) * np.arcsin(arg)
    else:
        result.duration = duration
    result.duration_hours = result.duration * 24.0

    # Limb darkening
    result.u1 = config.DEFAULT_LIMB_DARKENING_COEFFS[0]
    result.u2 = config.DEFAULT_LIMB_DARKENING_COEFFS[1]

    # Goodness of fit
    dof = len(time) - 4  # 4 fitted parameters
    result.chi_squared = best_chi2
    result.reduced_chi_squared = best_chi2 / max(dof, 1)
    result.bic = best_chi2 + 4 * np.log(len(time))

    result.model_flux = model_flux

    # --- Uncertainties from bootstrap ---
    if param_samples is not None and len(param_samples) > 5:
        result.rp_rs_err = np.std(param_samples[:, 0])
        result.a_rs_err = np.std(param_samples[:, 1])
        result.inclination_err = np.std(param_samples[:, 2])
        result.epoch_err = np.std(param_samples[:, 3])
        result.depth_err = 2 * rp_rs * result.rp_rs_err  # Error propagation
        result.depth_ppm_err = result.depth_err * 1e6
        result.duration_err = result.duration * 0.1  # Approximate
        result.duration_hours_err = result.duration_err * 24.0
        result.impact_param_err = np.sqrt(
            (np.cos(np.radians(inc)) * result.a_rs_err) ** 2 +
            (a_rs * np.sin(np.radians(inc)) * np.radians(result.inclination_err)) ** 2
        )
    else:
        # Rough uncertainty estimates
        result.rp_rs_err = rp_rs * 0.1
        result.depth_err = result.depth * 0.2
        result.depth_ppm_err = result.depth_err * 1e6
        result.duration_err = result.duration * 0.15
        result.duration_hours_err = result.duration_err * 24.0

    logger.info(f"Transit fit complete: depth={result.depth_ppm:.0f} ppm, "
                f"period={result.period:.4f} d, duration={result.duration_hours:.2f} h")

    return result


def _bootstrap_uncertainties(time, flux, flux_err, period, best_params,
                             n_bootstrap=100):
    """Estimate parameter uncertainties via bootstrap resampling."""
    n_data = len(time)
    samples = []

    for i in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.randint(0, n_data, size=n_data)
        t_boot = time[idx]
        f_boot = flux[idx] + np.random.normal(0, flux_err[idx])
        fe_boot = flux_err[idx]

        # Sort by time for proper model evaluation
        sort_idx = np.argsort(t_boot)
        t_boot, f_boot, fe_boot = t_boot[sort_idx], f_boot[sort_idx], fe_boot[sort_idx]

        try:
            result = minimize(
                _chi_squared, best_params,
                args=(t_boot, f_boot, fe_boot, period, True),
                method="Nelder-Mead",
                options={"maxiter": 1000}
            )
            if result.success or result.fun < 1e9:
                samples.append(result.x)
        except Exception:
            continue

    if len(samples) > 5:
        return np.array(samples)
    return None


def _quick_estimate(period, epoch, depth, duration):
    """Quick parameter estimate without fitting (used as fallback)."""
    result = TransitParams()
    result.period = period
    result.epoch = epoch
    result.depth = depth
    result.depth_ppm = depth * 1e6
    result.rp_rs = np.sqrt(max(depth, 0))
    result.duration = duration
    result.duration_hours = duration * 24.0

    # Large uncertainties for unfitted values
    result.depth_err = depth * 0.5
    result.depth_ppm_err = result.depth_err * 1e6
    result.rp_rs_err = result.rp_rs * 0.25
    result.duration_err = duration * 0.3
    result.duration_hours_err = result.duration_err * 24.0
    result.a_rs = 10.0
    result.inclination = 89.0

    return result



def fit_transit_mcmc(time: np.ndarray, flux: np.ndarray,
                     flux_err: np.ndarray, period: float,
                     initial_params: TransitParams) -> TransitParams:
    """Refine transit parameters using MCMC sampling with emcee.

    This provides more robust uncertainty estimates but is slower.
    """
    if not HAS_EMCEE:
        logger.warning("emcee not installed. Skipping MCMC refinement.")
        return initial_params

    def log_prior(theta):
        rp_rs, a_rs, inc, epoch = theta
        if 0.001 < rp_rs < 0.4 and 2 < a_rs < 100 and 70 < inc < 90:
            if abs(epoch - initial_params.epoch) < 0.1 * period:
                return 0.0
        return -np.inf

    def log_likelihood(theta):
        rp_rs, a_rs, inc, epoch = theta
        model = _batman_model(time, period, epoch, rp_rs, a_rs, inc)
        residuals = flux - model
        return -0.5 * np.sum((residuals / flux_err) ** 2 + np.log(2 * np.pi * flux_err ** 2))

    def log_posterior(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)

    # Initialize walkers
    p0 = np.array([initial_params.rp_rs, initial_params.a_rs,
                    initial_params.inclination, initial_params.epoch])
    ndim = len(p0)
    nwalkers = config.MCMC_NWALKERS
    pos = p0 + 1e-4 * np.random.randn(nwalkers, ndim)

    # Run MCMC
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
    logger.info(f"Running MCMC: {config.MCMC_NSTEPS} steps, {nwalkers} walkers")
    sampler.run_mcmc(pos, config.MCMC_NSTEPS, progress=True)

    # Extract results
    flat_samples = sampler.get_chain(discard=config.MCMC_BURNIN, flat=True)

    # Update parameters with MCMC posteriors
    result = TransitParams()
    result.period = period
    result.rp_rs = np.median(flat_samples[:, 0])
    result.rp_rs_err = np.std(flat_samples[:, 0])
    result.a_rs = np.median(flat_samples[:, 1])
    result.a_rs_err = np.std(flat_samples[:, 1])
    result.inclination = np.median(flat_samples[:, 2])
    result.inclination_err = np.std(flat_samples[:, 2])
    result.epoch = np.median(flat_samples[:, 3])
    result.epoch_err = np.std(flat_samples[:, 3])

    result.depth = result.rp_rs ** 2
    result.depth_err = 2 * result.rp_rs * result.rp_rs_err
    result.depth_ppm = result.depth * 1e6
    result.depth_ppm_err = result.depth_err * 1e6
    result.impact_param = result.a_rs * np.cos(np.radians(result.inclination))

    # Duration
    b = result.impact_param
    sin_i = np.sin(np.radians(result.inclination))
    arg = np.sqrt(max((1 + result.rp_rs) ** 2 - b ** 2, 0)) / (result.a_rs * sin_i)
    if 0 < arg <= 1:
        result.duration = (period / np.pi) * np.arcsin(arg)
    else:
        result.duration = initial_params.duration
    result.duration_hours = result.duration * 24.0
    result.duration_err = initial_params.duration_err
    result.duration_hours_err = result.duration_err * 24.0

    # Best-fit model
    result.model_flux = _batman_model(time, period, result.epoch,
                                       result.rp_rs, result.a_rs, result.inclination)

    # Goodness of fit
    residuals = flux - result.model_flux
    result.chi_squared = np.sum((residuals / flux_err) ** 2)
    result.reduced_chi_squared = result.chi_squared / max(len(time) - ndim, 1)

    logger.info(f"MCMC fit: Rp/Rs = {result.rp_rs:.4f} ± {result.rp_rs_err:.4f}")
    return result



if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    t = np.linspace(0, 10, 5000)
    period, epoch, rp_rs_true = 3.0, 1.5, 0.1

    if HAS_BATMAN:
        true_flux = _batman_model(t, period, epoch, rp_rs_true, 15.0, 88.0)
    else:
        true_flux = _analytic_transit_model(t, period, epoch, rp_rs_true, 15.0, 88.0)

    noise = np.random.normal(0, 0.001, len(t))
    flux = true_flux + noise
    flux_err = np.full_like(flux, 0.001)

    result = fit_transit(t, flux, flux_err, period=period, epoch=epoch,
                         depth=rp_rs_true ** 2, duration=0.15)
    print(result.summary())
