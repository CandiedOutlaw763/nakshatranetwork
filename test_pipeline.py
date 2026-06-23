"""Quick functional test suite for the exoplanet pipeline."""
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '.')
import numpy as np
from pathlib import Path

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name}")
        failed += 1

# === TEST 1: Transit Fitting ===
print("\n=== TEST 1: Transit Model Fitting ===")
from src.transit_fitting import fit_transit, _batman_model
np.random.seed(42)
t = np.linspace(0, 10, 5000)
period, epoch, rp_rs_true = 3.0, 1.5, 0.1
true_flux = _batman_model(t, period, epoch, rp_rs_true, 15.0, 88.0)
noise = np.random.normal(0, 0.001, len(t))
flux = true_flux + noise
flux_err = np.full_like(flux, 0.001)
result = fit_transit(t, flux, flux_err, period=period, epoch=epoch,
                     depth=rp_rs_true**2, duration=0.15)
print(f"  True Rp/Rs: {rp_rs_true:.4f} | Fitted: {result.rp_rs:.4f} +/- {result.rp_rs_err:.4f}")
print(f"  Depth: {result.depth_ppm:.0f} ppm | Duration: {result.duration_hours:.2f} h")
print(f"  Chi2_red: {result.reduced_chi_squared:.3f}")
check("Rp/Rs recovery", abs(result.rp_rs - rp_rs_true) < 0.02)

# === TEST 2: SNR Calculator ===
print("\n=== TEST 2: SNR & Confidence ===")
from src.snr_calculator import compute_transit_snr, compute_full_metrics
snr = compute_transit_snr(t, flux, flux_err, period, epoch, rp_rs_true**2, 0.15)
print(f"  Transit SNR: {snr['transit_snr']:.1f}")
print(f"  Single transit SNR: {snr['single_transit_snr']:.1f}")
print(f"  N transits: {snr['n_transits']}")
check("SNR > 5", snr['transit_snr'] > 5)

metrics = compute_full_metrics(t, flux, flux_err, period, epoch, rp_rs_true**2, 0.15,
                                sde=12.0, classifier_prob=0.88, classifier_class="PLANET")
print(f"  Combined confidence: {metrics.combined_confidence:.2f} ({metrics.confidence_label})")
print(f"  FAP: {metrics.false_alarm_prob:.2e}")
check("Confidence > 0.5", metrics.combined_confidence > 0.5)

# === TEST 3: BLS Detection ===
print("\n=== TEST 3: BLS Signal Detection ===")
from src.signal_detection import run_bls
det = run_bls(t, flux, flux_err)
print(f"  Detected period: {det.period:.4f} d (true: {period:.4f})")
print(f"  SDE: {det.sde:.1f}")
print(f"  Depth: {det.depth:.5f}")
check("Period recovery", abs(det.period - period) < 0.1)

# === TEST 4: Feature Extraction ===
print("\n=== TEST 4: Feature Extraction ===")
from src.feature_extraction import create_global_view, create_local_view
gv = create_global_view(t, flux, period, epoch)
lv = create_local_view(t, flux, period, epoch, 0.15)
print(f"  Global view shape: {gv.shape}")
print(f"  Local view shape: {lv.shape}")
check("Global view 201 bins", gv.shape[0] == 201)
check("Local view 61 bins", lv.shape[0] == 61)

# === TEST 5: CNN Model ===
print("\n=== TEST 5: CNN Model Architecture ===")
from src.classifier import build_cnn_model
model = build_cnn_model()
print(f"  Model parameters: {model.count_params():,}")
pred = model.predict([gv.reshape(1, -1, 1), lv.reshape(1, -1, 1)], verbose=0)
print(f"  Prediction shape: {pred.shape}")
print(f"  Probabilities: {pred[0]}")
check("CNN outputs 4 classes", pred.shape[1] == 4)
check("Probabilities sum to 1", abs(pred[0].sum() - 1.0) < 0.01)

# === TEST 6: PDF Report ===
print("\n=== TEST 6: PDF Report Generation ===")
from src.report_generator import generate_report
report_path = Path("results/test_report.pdf")
generate_report([], report_path)
check("PDF report created", report_path.exists())
if report_path.exists():
    print(f"  Report size: {report_path.stat().st_size:,} bytes")

# === SUMMARY ===
print("\n" + "=" * 50)
print(f"  RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
print("=" * 50)
