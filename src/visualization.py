"""
Visualization Module for the Exoplanet Detection Pipeline.

Provides publication-quality plotting functions using matplotlib (static)
and plotly (interactive) for light curves, periodograms, phase-folded
data, classification results, and parameter summaries.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.utils import logger, phase_fold, bin_data

# Apply dark style
plt.style.use(config.PLOT_STYLE)

# Custom colors
C = config.COLORS


def _setup_ax(ax, title="", xlabel="", ylabel=""):
    """Apply consistent styling to axes."""
    ax.set_title(title, fontsize=13, fontweight="bold",
                 color=C["text_primary"], pad=10)
    ax.set_xlabel(xlabel, fontsize=11, color=C["text_secondary"])
    ax.set_ylabel(ylabel, fontsize=11, color=C["text_secondary"])
    ax.tick_params(colors=C["text_secondary"], labelsize=9)
    ax.set_facecolor(C["bg_card"])
    for spine in ax.spines.values():
        spine.set_color(C["grid"])



def plot_raw_lightcurve(time: np.ndarray, flux: np.ndarray,
                        flux_err: np.ndarray = None,
                        tic_id: str = "",
                        save_path: Optional[Path] = None) -> plt.Figure:
    """Plot raw light curve (time vs flux).

    Args:
        time: Time array (BJD or TBJD)
        flux: Flux array
        flux_err: Flux errors (optional, shown as error bars)
        tic_id: TIC ID for title
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(C["bg_primary"])

    ax.scatter(time, flux, s=0.8, alpha=0.6, color=C["accent_blue"],
               edgecolors="none", rasterized=True)

    if flux_err is not None:
        ax.fill_between(time, flux - flux_err, flux + flux_err,
                        alpha=0.1, color=C["accent_blue"])

    _setup_ax(ax,
              title=f"Raw Light Curve — TIC {tic_id}" if tic_id else "Raw Light Curve",
              xlabel="Time (TBJD)", ylabel="Normalized Flux")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight",
                    facecolor=C["bg_primary"])
        logger.info(f"Saved plot: {save_path}")
    return fig


def plot_detrended_lightcurve(time: np.ndarray, flux: np.ndarray,
                               trend: np.ndarray = None,
                               transit_times: List[float] = None,
                               tic_id: str = "",
                               save_path: Optional[Path] = None) -> plt.Figure:
    """Plot detrended light curve with optional trend and transit markers.

    Args:
        time: Time array
        flux: Detrended flux
        trend: Original trend that was removed (shown in separate panel)
        transit_times: List of mid-transit times to mark
        tic_id: TIC ID for title
        save_path: Save path

    Returns:
        matplotlib Figure
    """
    n_panels = 2 if trend is not None else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels),
                             sharex=True)
    fig.patch.set_facecolor(C["bg_primary"])

    if n_panels == 1:
        axes = [axes]

    # Top panel: detrended flux
    ax = axes[0]
    ax.scatter(time, flux, s=0.8, alpha=0.6, color=C["accent_blue"],
               edgecolors="none", rasterized=True)

    if transit_times:
        for tt in transit_times:
            ax.axvline(tt, color=C["accent_green"], alpha=0.5, lw=0.8, ls="--")

    _setup_ax(ax,
              title=f"Detrended Light Curve — TIC {tic_id}" if tic_id else "Detrended Light Curve",
              xlabel="" if n_panels > 1 else "Time (TBJD)",
              ylabel="Relative Flux")

    # Bottom panel: trend
    if trend is not None:
        ax2 = axes[1]
        ax2.plot(time, trend, color=C["accent_orange"], lw=1.5, alpha=0.8)
        _setup_ax(ax2, title="Stellar Variability Trend",
                  xlabel="Time (TBJD)", ylabel="Trend Flux")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight",
                    facecolor=C["bg_primary"])
    return fig



def plot_periodogram(periods: np.ndarray, power: np.ndarray,
                     best_period: float = None,
                     sde: float = None,
                     method: str = "BLS",
                     tic_id: str = "",
                     save_path: Optional[Path] = None) -> plt.Figure:
    """Plot BLS/TLS periodogram (power vs period).

    Args:
        periods: Period grid
        power: Periodogram power values
        best_period: Best period to highlight
        sde: Signal Detection Efficiency
        method: 'BLS' or 'TLS'
        tic_id: TIC ID
        save_path: Save path

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(C["bg_primary"])

    ax.plot(periods, power, color=C["accent_purple"], lw=0.8, alpha=0.8)
    ax.fill_between(periods, 0, power, alpha=0.15, color=C["accent_purple"])

    if best_period is not None:
        idx = np.argmin(np.abs(periods - best_period))
        ax.axvline(best_period, color=C["accent_green"], lw=2, ls="--", alpha=0.8)
        ax.annotate(f"P = {best_period:.4f} d",
                    xy=(best_period, power[idx]),
                    xytext=(20, 20), textcoords="offset points",
                    fontsize=11, fontweight="bold", color=C["accent_green"],
                    arrowprops=dict(arrowstyle="->", color=C["accent_green"]),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=C["bg_card"],
                             edgecolor=C["accent_green"], alpha=0.9))

    sde_text = f"SDE = {sde:.1f}" if sde else ""
    _setup_ax(ax,
              title=f"{method} Periodogram — TIC {tic_id}  {sde_text}",
              xlabel="Period (days)", ylabel=f"{method} Power")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight",
                    facecolor=C["bg_primary"])
    return fig



def plot_phase_folded(time: np.ndarray, flux: np.ndarray,
                      period: float, epoch: float,
                      model_flux: np.ndarray = None,
                      duration: float = None,
                      n_bins: int = 50,
                      tic_id: str = "",
                      classification: str = None,
                      save_path: Optional[Path] = None) -> plt.Figure:
    """Plot phase-folded light curve with optional transit model overlay.

    Args:
        time: Time array
        flux: Normalized flux
        period: Folding period
        epoch: Folding epoch
        model_flux: Best-fit transit model
        duration: Transit duration (for shading)
        n_bins: Number of bins for binned curve
        tic_id: TIC ID
        classification: Classification label
        save_path: Save path

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                              gridspec_kw={"width_ratios": [3, 1.5]})
    fig.patch.set_facecolor(C["bg_primary"])

    phase = phase_fold(time, period, epoch)

    # --- Full phase view ---
    ax1 = axes[0]
    ax1.scatter(phase, flux, s=1.0, alpha=0.3, color=C["text_secondary"],
                edgecolors="none", rasterized=True, label="Data")

    # Binned data
    bin_centers, bin_means, bin_stds = bin_data(phase, flux, n_bins,
                                                x_range=(-0.5, 0.5))
    valid = np.isfinite(bin_means)
    ax1.scatter(bin_centers[valid], bin_means[valid], s=25, color=C["accent_blue"],
                zorder=5, edgecolors="white", linewidths=0.5, label="Binned")
    ax1.errorbar(bin_centers[valid], bin_means[valid], yerr=bin_stds[valid],
                 fmt="none", ecolor=C["accent_blue"], alpha=0.4, zorder=4)

    # Transit model
    if model_flux is not None:
        model_phase = phase_fold(time, period, epoch)
        sort_idx = np.argsort(model_phase)
        ax1.plot(model_phase[sort_idx], model_flux[sort_idx],
                 color=C["accent_green"], lw=2.5, alpha=0.9, label="Transit Model",
                 path_effects=[pe.Stroke(linewidth=4, foreground=C["bg_card"]),
                              pe.Normal()])

    # Transit duration shading
    if duration:
        half_dur = (duration / period) / 2.0
        ax1.axvspan(-half_dur, half_dur, alpha=0.08, color=C["accent_green"])

    cls_color = {
        "PLANET": C["planet"], "ECLIPSING_BINARY": C["eb"],
        "BLEND": C["blend"], "OTHER": C["other"]
    }.get(classification, C["text_secondary"])

    title = f"Phase-Folded — TIC {tic_id} (P = {period:.4f} d)"
    if classification:
        title += f"  •  {classification}"
    _setup_ax(ax1, title=title, xlabel="Phase", ylabel="Relative Flux")
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.8,
               facecolor=C["bg_card"], edgecolor=C["grid"])

    # --- Zoomed transit view ---
    ax2 = axes[1]
    zoom = 0.08
    zoom_mask = np.abs(phase) < zoom

    if np.sum(zoom_mask) > 5:
        ax2.scatter(phase[zoom_mask], flux[zoom_mask], s=3, alpha=0.4,
                    color=C["text_secondary"], edgecolors="none")

        # Binned zoomed
        bc, bm, bs = bin_data(phase[zoom_mask], flux[zoom_mask], 30,
                              x_range=(-zoom, zoom))
        v = np.isfinite(bm)
        ax2.scatter(bc[v], bm[v], s=30, color=cls_color,
                    zorder=5, edgecolors="white", linewidths=0.5)
        ax2.errorbar(bc[v], bm[v], yerr=bs[v], fmt="none",
                     ecolor=cls_color, alpha=0.4)

        if model_flux is not None:
            zm = zoom_mask
            si = np.argsort(phase[zm])
            ax2.plot(phase[zm][si], model_flux[zm][si],
                     color=C["accent_green"], lw=2.5, alpha=0.9)

    _setup_ax(ax2, title="Transit Detail", xlabel="Phase", ylabel="")
    ax2.set_xlim(-zoom, zoom)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight",
                    facecolor=C["bg_primary"])
    return fig



def plot_classification(class_probs: Dict[str, float],
                        tic_id: str = "",
                        save_path: Optional[Path] = None) -> plt.Figure:
    """Plot classification probabilities as a horizontal bar chart.

    Args:
        class_probs: Dictionary mapping class name to probability
        tic_id: TIC ID
        save_path: Save path

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(C["bg_primary"])

    classes = list(class_probs.keys())
    probs = list(class_probs.values())
    colors = [C.get(c.lower(), C["text_secondary"]) for c in classes]
    color_map = {"PLANET": C["planet"], "ECLIPSING_BINARY": C["eb"],
                 "BLEND": C["blend"], "OTHER": C["other"]}
    colors = [color_map.get(c, C["text_secondary"]) for c in classes]

    bars = ax.barh(classes, probs, color=colors, alpha=0.85, edgecolor="none",
                   height=0.6)

    # Value labels
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{prob:.1%}", va="center", ha="left",
                fontsize=11, fontweight="bold", color=C["text_primary"])

    ax.set_xlim(0, 1.15)
    _setup_ax(ax,
              title=f"Classification — TIC {tic_id}" if tic_id else "Classification",
              xlabel="Probability", ylabel="")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight",
                    facecolor=C["bg_primary"])
    return fig



def plot_parameter_summary(params: Dict,
                           confidence: Dict = None,
                           tic_id: str = "",
                           save_path: Optional[Path] = None) -> plt.Figure:
    """Create a parameter summary card as a styled table figure.

    Args:
        params: Transit parameter dictionary
        confidence: Confidence metrics dictionary
        tic_id: TIC ID
        save_path: Save path

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(C["bg_primary"])
    ax.set_facecolor(C["bg_card"])
    ax.axis("off")

    # Title
    ax.text(0.5, 0.95, f"Transit Parameters — TIC {tic_id}",
            transform=ax.transAxes, fontsize=14, fontweight="bold",
            color=C["accent_blue"], ha="center", va="top")

    # Parameter table
    rows = [
        ("Orbital Period", f"{params.get('period', 0):.6f} ± {params.get('period_err', 0):.6f} days"),
        ("Transit Depth", f"{params.get('depth_ppm', 0):.1f} ± {params.get('depth_ppm_err', 0):.1f} ppm"),
        ("Transit Duration", f"{params.get('duration_hours', 0):.2f} ± {params.get('duration_hours_err', 0):.2f} hours"),
        ("Rp/Rs", f"{params.get('rp_rs', 0):.4f} ± {params.get('rp_rs_err', 0):.4f}"),
        ("a/Rs", f"{params.get('a_rs', 0):.2f} ± {params.get('a_rs_err', 0):.2f}"),
        ("Inclination", f"{params.get('inclination', 0):.2f} ± {params.get('inclination_err', 0):.2f}°"),
        ("Impact Parameter", f"{params.get('impact_param', 0):.3f} ± {params.get('impact_param_err', 0):.3f}"),
        ("χ²_red", f"{params.get('reduced_chi_squared', 0):.3f}"),
    ]

    if confidence:
        rows.extend([
            ("", ""),  # Spacer
            ("Transit SNR", f"{confidence.get('transit_snr', 0):.1f}"),
            ("BLS SDE", f"{confidence.get('bls_sde', 0):.1f}"),
            ("FAP", f"{confidence.get('false_alarm_prob', 0):.2e}"),
            ("Confidence", f"{confidence.get('combined_confidence', 0):.1%}"),
        ])

    y_start = 0.85
    for i, (label, value) in enumerate(rows):
        y = y_start - i * 0.058
        if label == "":
            continue
        ax.text(0.15, y, label, transform=ax.transAxes, fontsize=10,
                color=C["text_secondary"], ha="left", va="center",
                fontfamily="monospace")
        ax.text(0.85, y, value, transform=ax.transAxes, fontsize=10,
                color=C["text_primary"], ha="right", va="center",
                fontweight="bold", fontfamily="monospace")
        # Separator line
        ax.axhline(y=y - 0.015, xmin=0.1, xmax=0.9,
                   transform=ax.transAxes, color=C["grid"], lw=0.5, alpha=0.5)

    if save_path:
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight",
                    facecolor=C["bg_primary"])
    return fig



def plot_candidate_summary(time: np.ndarray, flux: np.ndarray,
                           period: float, epoch: float,
                           periods: np.ndarray = None,
                           power: np.ndarray = None,
                           model_flux: np.ndarray = None,
                           duration: float = None,
                           class_probs: Dict[str, float] = None,
                           params: Dict = None,
                           confidence: Dict = None,
                           sde: float = None,
                           tic_id: str = "",
                           classification: str = None,
                           save_path: Optional[Path] = None) -> plt.Figure:
    """Create a comprehensive 4-panel summary for a candidate.

    Panels:
    1. Raw/detrended light curve with transits marked
    2. BLS/TLS periodogram
    3. Phase-folded light curve with model
    4. Classification probabilities + parameter summary

    Args:
        time, flux: Light curve data
        period, epoch: Detected period/epoch
        periods, power: Periodogram data
        model_flux: Best-fit transit model
        duration: Transit duration
        class_probs: Classification probabilities
        params: Transit parameters dict
        confidence: Confidence metrics dict
        sde: Signal Detection Efficiency
        tic_id: TIC ID
        classification: Predicted class
        save_path: Save path

    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor(C["bg_primary"])

    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3,
                  height_ratios=[1, 1, 1])

    # --- Panel 1: Light curve ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.scatter(time, flux, s=0.8, alpha=0.5, color=C["accent_blue"],
                edgecolors="none", rasterized=True)

    # Mark transit times
    if period and epoch:
        t_min, t_max = np.nanmin(time), np.nanmax(time)
        transit_times = epoch + np.arange(
            -int((epoch - t_min) / period) - 1,
            int((t_max - epoch) / period) + 2
        ) * period
        for tt in transit_times:
            if t_min <= tt <= t_max:
                ax1.axvline(tt, color=C["accent_green"], alpha=0.3, lw=0.8, ls="--")

    _setup_ax(ax1, title=f"Light Curve — TIC {tic_id}", xlabel="Time (TBJD)",
              ylabel="Relative Flux")

    # --- Panel 2: Periodogram ---
    ax2 = fig.add_subplot(gs[1, 0])
    if periods is not None and power is not None:
        ax2.plot(periods, power, color=C["accent_purple"], lw=0.8)
        ax2.fill_between(periods, 0, power, alpha=0.15, color=C["accent_purple"])
        if period:
            ax2.axvline(period, color=C["accent_green"], lw=2, ls="--")
            ax2.set_title(f"Periodogram  (P = {period:.4f} d, SDE = {sde:.1f})" if sde else
                         f"Periodogram  (P = {period:.4f} d)",
                         fontsize=12, color=C["text_primary"], fontweight="bold")
    _setup_ax(ax2, xlabel="Period (days)", ylabel="Power")

    # --- Panel 3: Phase-folded ---
    ax3 = fig.add_subplot(gs[1, 1])
    if period and epoch:
        phase = phase_fold(time, period, epoch)
        ax3.scatter(phase, flux, s=1, alpha=0.3, color=C["text_secondary"],
                    edgecolors="none", rasterized=True)

        bc, bm, bs = bin_data(phase, flux, 50, x_range=(-0.5, 0.5))
        v = np.isfinite(bm)
        cls_color = {"PLANET": C["planet"], "ECLIPSING_BINARY": C["eb"],
                     "BLEND": C["blend"]}.get(classification, C["accent_blue"])
        ax3.scatter(bc[v], bm[v], s=20, color=cls_color, zorder=5,
                    edgecolors="white", linewidths=0.4)

        if model_flux is not None:
            si = np.argsort(phase)
            ax3.plot(phase[si], model_flux[si], color=C["accent_green"],
                     lw=2.5, alpha=0.9,
                     path_effects=[pe.Stroke(linewidth=4, foreground=C["bg_card"]),
                                  pe.Normal()])

    _setup_ax(ax3, title="Phase-Folded Light Curve",
              xlabel="Phase", ylabel="Relative Flux")

    # --- Panel 4: Classification + Parameters ---
    ax4 = fig.add_subplot(gs[2, 0])
    if class_probs:
        classes = list(class_probs.keys())
        probs = list(class_probs.values())
        color_map = {"PLANET": C["planet"], "ECLIPSING_BINARY": C["eb"],
                     "BLEND": C["blend"], "OTHER": C["other"]}
        colors = [color_map.get(c, C["text_secondary"]) for c in classes]
        ax4.barh(classes, probs, color=colors, alpha=0.85, height=0.6)
        for i, (c, p) in enumerate(zip(classes, probs)):
            ax4.text(p + 0.02, i, f"{p:.1%}", va="center",
                     fontsize=10, fontweight="bold", color=C["text_primary"])
        ax4.set_xlim(0, 1.15)
    _setup_ax(ax4, title="Classification", xlabel="Probability", ylabel="")

    # --- Panel 5: Parameters ---
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")
    ax5.set_facecolor(C["bg_card"])

    if params:
        text_lines = [
            f"Period:     {params.get('period', 0):.6f} ± {params.get('period_err', 0):.6f} d",
            f"Depth:      {params.get('depth_ppm', 0):.0f} ± {params.get('depth_ppm_err', 0):.0f} ppm",
            f"Duration:   {params.get('duration_hours', 0):.2f} ± {params.get('duration_hours_err', 0):.2f} h",
            f"Rp/Rs:      {params.get('rp_rs', 0):.4f} ± {params.get('rp_rs_err', 0):.4f}",
            f"a/Rs:       {params.get('a_rs', 0):.1f} ± {params.get('a_rs_err', 0):.1f}",
            f"Inc:        {params.get('inclination', 0):.1f} ± {params.get('inclination_err', 0):.1f}°",
        ]
        if confidence:
            text_lines.extend([
                "",
                f"SNR:        {confidence.get('transit_snr', 0):.1f}",
                f"SDE:        {confidence.get('bls_sde', 0):.1f}",
                f"FAP:        {confidence.get('false_alarm_prob', 0):.2e}",
                f"Confidence: {confidence.get('combined_confidence', 0):.1%}",
            ])

        text = "\n".join(text_lines)
        ax5.text(0.1, 0.9, text, transform=ax5.transAxes,
                 fontsize=10, fontfamily="monospace",
                 color=C["text_primary"], va="top",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor=C["bg_card"],
                          edgecolor=C["accent_blue"], alpha=0.9))
        ax5.set_title("Transit Parameters", fontsize=12,
                       color=C["text_primary"], fontweight="bold")

    # Supertitle
    label = classification or "Unknown"
    fig.suptitle(f"Candidate Summary — TIC {tic_id}  •  {label}",
                 fontsize=16, fontweight="bold", color=C["accent_blue"],
                 y=0.98)

    if save_path:
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight",
                    facecolor=C["bg_primary"])
        logger.info(f"Saved summary plot: {save_path}")
    return fig



def plot_training_history(history: Dict,
                          save_path: Optional[Path] = None) -> plt.Figure:
    """Plot CNN training history (loss and accuracy curves).

    Args:
        history: Keras training history dictionary
        save_path: Save path

    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(C["bg_primary"])

    # Loss
    ax1.plot(history.get("loss", []), color=C["accent_blue"], lw=2, label="Train")
    ax1.plot(history.get("val_loss", []), color=C["accent_orange"], lw=2, label="Val")
    _setup_ax(ax1, title="Training Loss", xlabel="Epoch", ylabel="Loss")
    ax1.legend(framealpha=0.8, facecolor=C["bg_card"], edgecolor=C["grid"])

    # Accuracy
    ax2.plot(history.get("accuracy", []), color=C["accent_blue"], lw=2, label="Train")
    ax2.plot(history.get("val_accuracy", []), color=C["accent_orange"], lw=2, label="Val")
    _setup_ax(ax2, title="Training Accuracy", xlabel="Epoch", ylabel="Accuracy")
    ax2.legend(framealpha=0.8, facecolor=C["bg_card"], edgecolor=C["grid"])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight",
                    facecolor=C["bg_primary"])
    return fig


def plot_confusion_matrix(cm: np.ndarray,
                          classes: List[str] = None,
                          save_path: Optional[Path] = None) -> plt.Figure:
    """Plot confusion matrix heatmap.

    Args:
        cm: Confusion matrix (N×N array)
        classes: Class labels
        save_path: Save path

    Returns:
        matplotlib Figure
    """
    if classes is None:
        classes = config.CLASSIFICATION_CLASSES

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(C["bg_primary"])

    # Normalize
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="magma", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Labels
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm_norm[i, j] > 0.5 else C["text_primary"]
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.0%})",
                    ha="center", va="center", fontsize=9, color=color)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9,
                       color=C["text_secondary"])
    ax.set_yticklabels(classes, fontsize=9, color=C["text_secondary"])

    _setup_ax(ax, title="Confusion Matrix", xlabel="Predicted", ylabel="True")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight",
                    facecolor=C["bg_primary"])
    return fig



def create_plotly_lightcurve(time, flux, flux_err=None, tic_id=""):
    """Create interactive plotly light curve figure."""
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=time, y=flux, mode="markers",
        marker=dict(size=2, color=C["accent_blue"], opacity=0.5),
        name="Flux",
        hovertemplate="Time: %{x:.4f}<br>Flux: %{y:.6f}<extra></extra>"
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=C["bg_primary"],
        plot_bgcolor=C["bg_card"],
        title=dict(text=f"Light Curve — TIC {tic_id}", font=dict(color=C["accent_blue"])),
        xaxis=dict(title="Time (TBJD)", gridcolor=C["grid"]),
        yaxis=dict(title="Relative Flux", gridcolor=C["grid"]),
        font=dict(color=C["text_primary"]),
        margin=dict(l=60, r=30, t=50, b=50),
    )
    return fig


def create_plotly_periodogram(periods, power, best_period=None, sde=None):
    """Create interactive plotly periodogram figure."""
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=periods, y=power, mode="lines",
        line=dict(color=C["accent_purple"], width=1),
        fill="tozeroy", fillcolor="rgba(124, 58, 237, 0.15)",
        name="Power",
    ))

    if best_period:
        fig.add_vline(x=best_period, line_dash="dash",
                      line_color=C["accent_green"], line_width=2,
                      annotation_text=f"P={best_period:.4f} d",
                      annotation_font_color=C["accent_green"])

    title = "BLS Periodogram"
    if sde:
        title += f" (SDE = {sde:.1f})"

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=C["bg_primary"],
        plot_bgcolor=C["bg_card"],
        title=dict(text=title, font=dict(color=C["accent_blue"])),
        xaxis=dict(title="Period (days)", gridcolor=C["grid"]),
        yaxis=dict(title="Power", gridcolor=C["grid"]),
        font=dict(color=C["text_primary"]),
        margin=dict(l=60, r=30, t=50, b=50),
    )
    return fig


def create_plotly_phase_folded(phase, flux, bin_centers=None, bin_means=None,
                                model_phase=None, model_flux=None,
                                classification=None):
    """Create interactive plotly phase-folded figure."""
    import plotly.graph_objects as go

    cls_color = {"PLANET": C["planet"], "ECLIPSING_BINARY": C["eb"],
                 "BLEND": C["blend"]}.get(classification, C["accent_blue"])

    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=phase, y=flux, mode="markers",
        marker=dict(size=2, color=C["text_secondary"], opacity=0.3),
        name="Data",
    ))

    if bin_centers is not None and bin_means is not None:
        fig.add_trace(go.Scatter(
            x=bin_centers, y=bin_means, mode="markers",
            marker=dict(size=6, color=cls_color, line=dict(width=1, color="white")),
            name="Binned",
        ))

    if model_phase is not None and model_flux is not None:
        si = np.argsort(model_phase)
        fig.add_trace(go.Scatter(
            x=model_phase[si], y=model_flux[si], mode="lines",
            line=dict(color=C["accent_green"], width=3),
            name="Transit Model",
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=C["bg_primary"],
        plot_bgcolor=C["bg_card"],
        title=dict(text="Phase-Folded Light Curve", font=dict(color=C["accent_blue"])),
        xaxis=dict(title="Phase", gridcolor=C["grid"]),
        yaxis=dict(title="Relative Flux", gridcolor=C["grid"]),
        font=dict(color=C["text_primary"]),
        margin=dict(l=60, r=30, t=50, b=50),
    )
    return fig
