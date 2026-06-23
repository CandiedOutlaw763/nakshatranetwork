"""
Exoplanet Detection Pipeline — Main Orchestrator.

End-to-end CLI pipeline that ties together all modules:
download -> preprocess -> detect -> extract features -> classify -> fit -> report.

Usage:
    python pipeline.py run-all --sector 1 --sample 100
    python pipeline.py download --sector 1
    python pipeline.py preprocess
    python pipeline.py detect
    python pipeline.py classify
    python pipeline.py fit
    python pipeline.py report
"""

import sys
import os
import argparse
import time as _time
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

# Project imports
sys.path.insert(0, str(Path(__file__).parent))
import config
from src.utils import (logger, timer, save_pickle, load_pickle,
                       save_json, save_results_csv, ProgressTracker)



@timer
def stage_download(sector: int = None, sample: int = None):
    """Stage 1: Download TESS light curves and catalogs."""
    from src.data_acquisition import (download_tess_lightcurves,
                                       download_toi_catalog,
                                       download_eb_catalog,
                                       create_labeled_dataset,
                                       generate_synthetic_transits,
                                       generate_synthetic_ebs)

    sector = sector or config.TESS_SECTOR
    sample = sample or config.SAMPLE_SIZE

    logger.info(f"=== STAGE 1: DATA ACQUISITION ===")
    logger.info(f"Sector: {sector} | Sample: {sample or 'full sector'}")

    # Download catalogs first
    toi_df = download_toi_catalog()
    eb_df = download_eb_catalog()

    # Query MAST directly for all TIC IDs observed in this sector
    from src.data_acquisition import query_sector_tic_ids
    all_sector_tics = query_sector_tic_ids(sector)
    logger.info(f"MAST reports {len(all_sector_tics)} targets observed in Sector {sector}.")

    # Cross-match with TOI catalog to prioritize known planet candidates
    toi_tics = set()
    if "tid" in toi_df.columns:
        toi_tics = set(toi_df["tid"].dropna().astype(int).unique())
    
    # Put TOI targets first (most scientifically interesting), then the rest
    prioritized_tics = [t for t in all_sector_tics if t in toi_tics]
    remaining_tics = [t for t in all_sector_tics if t not in toi_tics]
    tic_list = prioritized_tics + remaining_tics
    logger.info(f"Prioritized {len(prioritized_tics)} TOI targets + {len(remaining_tics)} other targets.")

    # Download light curves
    saved_files = download_tess_lightcurves(sector=sector, sample_size=sample, tic_list=tic_list)

    from src.utils import tic_id_from_filename
    tic_ids = [
        tic_id_from_filename(p.name)
        for p in saved_files
        if tic_id_from_filename(p.name) is not None
    ]

    # Create labeled dataset
    if tic_ids:
        create_labeled_dataset(tic_ids, toi_df, eb_df)

    # Generate synthetic training data
    generate_synthetic_transits(n_samples=200)
    generate_synthetic_ebs(n_samples=200)

    logger.info(f"Download complete: {len(tic_ids) if tic_ids else 0} light curves")
    return tic_ids


@timer
def stage_preprocess():
    """Stage 2: Preprocess all downloaded light curves."""
    from src.preprocessing import preprocess_all

    logger.info(f"=== STAGE 2: PREPROCESSING ===")

    results = preprocess_all(
        raw_dir=config.RAW_DATA_DIR,
        output_dir=config.PROCESSED_DATA_DIR,
        n_jobs=4
    )

    n_processed = len(list(config.PROCESSED_DATA_DIR.glob("*_processed.pkl")))
    logger.info(f"Preprocessing complete: {n_processed} light curves processed")
    return n_processed


@timer
def stage_detect():
    """Stage 3: Run BLS/TLS signal detection on all preprocessed light curves."""
    from src.signal_detection import detect_signals

    logger.info(f"=== STAGE 3: SIGNAL DETECTION ===")

    processed_files = sorted(config.PROCESSED_DATA_DIR.glob("*_processed.pkl"))
    if not processed_files:
        logger.error("No preprocessed files found. Run preprocess stage first.")
        return []

    all_detections = []
    tracker = ProgressTracker(len(processed_files), "Detecting signals")

    for filepath in processed_files:
        try:
            data = load_pickle(filepath)
            t = data.get("time", np.array([]))
            f = data.get("flux", np.array([]))
            fe = data.get("flux_err", np.array([]))

            if len(t) < 100:
                tracker.update()
                continue

            # Extract TIC ID from filename
            from src.utils import tic_id_from_filename
            extracted = tic_id_from_filename(filepath.name)
            if extracted is None:
                continue
            tic_id = str(extracted)

            # Run detection
            method = "tls" if config.USE_TLS else "bls"
            detections = detect_signals(t, f, fe, method=method,
                                         max_signals=config.MAX_PLANETS_PER_STAR)

            for i, det in enumerate(detections):
                det_dict = {
                    "tic_id": tic_id,
                    "signal_num": i + 1,
                    "period": det.period,
                    "epoch": det.epoch,
                    "depth": det.depth,
                    "duration": det.duration,
                    "sde": det.sde,
                    "fap": det.fap,
                    "method": det.method,
                    "filepath": str(filepath),
                }
                all_detections.append(det_dict)

        except Exception as e:
            logger.warning(f"Detection failed for {filepath.name}: {e}")

        tracker.update()

    # Save detection results
    if all_detections:
        det_df = pd.DataFrame(all_detections)
        det_path = config.RESULTS_DIR / "detections.csv"
        det_df.to_csv(det_path, index=False)
        save_pickle(all_detections, config.RESULTS_DIR / "detections.pkl")
        logger.info(f"Detection complete: {len(all_detections)} signals in "
                     f"{len(set(d['tic_id'] for d in all_detections))} stars")

        # Summary
        sig_count = sum(1 for d in all_detections if d["sde"] >= config.SDE_THRESHOLD)
        logger.info(f"Significant detections (SDE >= {config.SDE_THRESHOLD}): {sig_count}")
    else:
        logger.warning("No detections found")

    return all_detections


@timer
def stage_extract_features(detections: List[Dict] = None):
    """Stage 4: Extract features for classification."""
    from src.feature_extraction import (create_global_view, create_local_view,
                                         extract_features)

    logger.info(f"=== STAGE 4: FEATURE EXTRACTION ===")

    # Load detections if not provided
    if detections is None:
        det_path = config.RESULTS_DIR / "detections.pkl"
        if det_path.exists():
            detections = load_pickle(det_path)
        else:
            logger.error("No detections file found. Run detect stage first.")
            return None

    # Filter significant detections
    sig_detections = [d for d in detections if d.get("sde", 0) >= config.SDE_THRESHOLD]
    if not sig_detections:
        sig_detections = detections[:50]  # Take top 50 anyway for analysis
        logger.warning("No detections above SDE threshold. Using top detections.")

    global_views = []
    local_views = []
    features_list = []
    valid_detections = []

    tracker = ProgressTracker(len(sig_detections), "Extracting features")

    for det in sig_detections:
        try:
            filepath = det.get("filepath")
            if not filepath or not Path(filepath).exists():
                tracker.update()
                continue

            data = load_pickle(filepath)
            t = data.get("time", np.array([]))
            f = data.get("flux", np.array([]))
            fe = data.get("flux_err", np.array([]))

            if len(t) < 50:
                tracker.update()
                continue

            period = det["period"]
            epoch = det["epoch"]
            duration = det.get("duration", 0.1)
            depth = det.get("depth", 0.001)

            # Create views
            gv = create_global_view(t, f, period, epoch)
            lv = create_local_view(t, f, period, epoch, duration)

            # Create a mock detection result for extract_features
            class MockDetection:
                pass
            mock = MockDetection()
            mock.period = period
            mock.epoch = epoch
            mock.depth = depth
            mock.duration = duration
            mock.sde = det.get("sde", 0)

            feats = extract_features(t, f, fe, mock)

            global_views.append(gv)
            local_views.append(lv)
            features_list.append(feats)
            valid_detections.append(det)

        except Exception as e:
            logger.warning(f"Feature extraction failed for TIC {det.get('tic_id', '?')}: {e}")

        tracker.update()

    if global_views:
        X_global = np.array(global_views)
        X_local = np.array(local_views)

        # Build feature matrix
        feature_names = list(features_list[0].keys()) if features_list else []
        X_features = np.array([[f.get(name, 0) for name in feature_names]
                                for f in features_list])
        X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)

        # Save
        np.save(config.PROCESSED_DATA_DIR / "X_global.npy", X_global)
        np.save(config.PROCESSED_DATA_DIR / "X_local.npy", X_local)
        np.save(config.PROCESSED_DATA_DIR / "X_features.npy", X_features)
        save_pickle(valid_detections, config.RESULTS_DIR / "valid_detections.pkl")
        save_pickle(feature_names, config.PROCESSED_DATA_DIR / "feature_names.pkl")

        logger.info(f"Feature extraction complete: {len(global_views)} candidates")
        logger.info(f"Global views: {X_global.shape}, Local views: {X_local.shape}, "
                     f"Features: {X_features.shape}")
    else:
        logger.warning("No features extracted")

    return valid_detections


@timer
def stage_classify(detections: List[Dict] = None):
    """Stage 5: Classify detected signals."""
    from src.classifier import load_models, classify

    logger.info(f"=== STAGE 5: CLASSIFICATION ===")

    # Load feature arrays
    gv_path = config.PROCESSED_DATA_DIR / "X_global.npy"
    lv_path = config.PROCESSED_DATA_DIR / "X_local.npy"
    feat_path = config.PROCESSED_DATA_DIR / "X_features.npy"

    if not all(p.exists() for p in [gv_path, lv_path, feat_path]):
        logger.error("Feature files not found. Run extract stage first.")
        return []

    X_global = np.load(gv_path)
    X_local = np.load(lv_path)
    X_features = np.load(feat_path)

    # Load detections
    if detections is None:
        det_path = config.RESULTS_DIR / "valid_detections.pkl"
        if det_path.exists():
            detections = load_pickle(det_path)
        else:
            logger.error("No valid detections file found.")
            return []

    # Try to load trained models
    try:
        cnn_model, rf_model = load_models(config.MODELS_DIR)
        logger.info("Loaded trained models")
    except Exception as e:
        logger.warning(f"Could not load trained models: {e}")
        logger.info("Using feature-based heuristic classification instead")
        return _heuristic_classify(detections, X_features)

    # Run classification
    results = []
    for i in range(len(X_global)):
        try:
            results_dict = classify(
                X_global[i:i+1], X_local[i:i+1], X_features[i:i+1],
                cnn_model, rf_model
            )
            label = results_dict["predicted_classes"][0]
            probs = results_dict["probabilities"]
            confidence = results_dict["confidence"][0]

            det = detections[i] if i < len(detections) else {}
            det["classification"] = label
            det["confidence"] = confidence
            det["class_probs"] = {
                cls: float(probs[0][j])
                for j, cls in enumerate(config.CLASSIFICATION_CLASSES)
            }
            results.append(det)

        except Exception as e:
            logger.warning(f"Classification failed for detection {i}: {e}")
            det = detections[i] if i < len(detections) else {}
            det["classification"] = "OTHER"
            det["confidence"] = 0.0
            results.append(det)

    # Save classification results
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != "class_probs"}
                                 for r in results])
    results_df.to_csv(config.RESULTS_DIR / "classifications.csv", index=False)
    save_pickle(results, config.RESULTS_DIR / "classifications.pkl")

    # Summary
    class_counts = {}
    for r in results:
        cls = r.get("classification", "OTHER")
        class_counts[cls] = class_counts.get(cls, 0) + 1
    logger.info(f"Classification summary: {class_counts}")

    return results


def _heuristic_classify(detections, X_features):
    """Heuristic classification when no trained model is available.

    Uses simple rules based on extracted features:
    - Deep transits (>5%) -> ECLIPSING_BINARY
    - V-shaped (>0.7) + deep -> ECLIPSING_BINARY
    - Odd/even ratio > 1.5 -> ECLIPSING_BINARY
    - Secondary eclipse > 0.5 × primary -> ECLIPSING_BINARY
    - Shallow + U-shaped -> PLANET
    - Low SDE -> OTHER
    """
    results = []
    feature_names = None
    fn_path = config.PROCESSED_DATA_DIR / "feature_names.pkl"
    if fn_path.exists():
        feature_names = load_pickle(fn_path)

    for i, det in enumerate(detections):
        feats = {}
        if feature_names and i < len(X_features):
            for j, name in enumerate(feature_names):
                feats[name] = X_features[i][j] if j < len(X_features[i]) else 0

        depth = det.get("depth", feats.get("transit_depth", 0))
        sde = det.get("sde", feats.get("sde", 0))
        v_shape = feats.get("v_shape", 0)
        odd_even = feats.get("odd_even_ratio", 1.0)
        secondary = feats.get("secondary_depth", 0)

        # Classification rules
        if depth > 0.05 or (v_shape > 0.7 and depth > 0.01):
            classification = "ECLIPSING_BINARY"
            confidence = min(0.9, 0.5 + depth * 5)
        elif odd_even > 1.5 or (secondary > 0.3 * depth and depth > 0.005):
            classification = "ECLIPSING_BINARY"
            confidence = 0.6
        elif sde < config.SDE_THRESHOLD * 0.7:
            classification = "OTHER"
            confidence = 0.3
        elif depth < 0.03 and v_shape < 0.5:
            classification = "PLANET"
            confidence = min(0.85, 0.3 + sde / 20.0)
        else:
            classification = "BLEND"
            confidence = 0.4

        probs = {cls: 0.05 for cls in config.CLASSIFICATION_CLASSES}
        probs[classification] = confidence
        remaining = 1.0 - confidence
        for cls in probs:
            if cls != classification:
                probs[cls] = remaining / (len(probs) - 1)

        det["classification"] = classification
        det["confidence"] = confidence
        det["class_probs"] = probs
        results.append(det)

    # Save
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != "class_probs"}
                                 for r in results])
    results_df.to_csv(config.RESULTS_DIR / "classifications.csv", index=False)
    save_pickle(results, config.RESULTS_DIR / "classifications.pkl")

    class_counts = {}
    for r in results:
        cls = r.get("classification", "OTHER")
        class_counts[cls] = class_counts.get(cls, 0) + 1
    logger.info(f"Heuristic classification summary: {class_counts}")

    return results


@timer
def stage_fit(classifications: List[Dict] = None):
    """Stage 6: Fit transit models to planet candidates."""
    from src.transit_fitting import fit_transit
    from src.snr_calculator import compute_full_metrics

    logger.info(f"=== STAGE 6: TRANSIT FITTING ===")

    # Load classifications
    if classifications is None:
        cls_path = config.RESULTS_DIR / "classifications.pkl"
        if cls_path.exists():
            classifications = load_pickle(cls_path)
        else:
            logger.error("No classification results found. Run classify stage first.")
            return []

    # Filter planet candidates (or fit all with SDE above threshold)
    candidates = [c for c in classifications
                  if c.get("classification") in ("PLANET", "ECLIPSING_BINARY")
                  or c.get("sde", 0) >= config.SDE_THRESHOLD]

    if not candidates:
        candidates = classifications[:20]  # Fit top 20 anyway
        logger.warning("No planet candidates found. Fitting top detections.")

    fitted_results = []
    tracker = ProgressTracker(len(candidates), "Fitting transits")

    for cand in candidates:
        try:
            filepath = cand.get("filepath")
            if not filepath or not Path(filepath).exists():
                tracker.update()
                continue

            data = load_pickle(filepath)
            t = data.get("time", np.array([]))
            f = data.get("flux", np.array([]))
            fe = data.get("flux_err", np.array([]))

            if len(t) < 50:
                tracker.update()
                continue

            # Fit transit model
            params = fit_transit(
                t, f, fe,
                period=cand["period"],
                epoch=cand["epoch"],
                depth=cand.get("depth", 0.001),
                duration=cand.get("duration", 0.1)
            )

            # Compute confidence metrics
            metrics = compute_full_metrics(
                t, f, fe,
                period=cand["period"],
                epoch=cand["epoch"],
                depth=params.depth,
                duration=params.duration,
                model_flux=params.model_flux,
                sde=cand.get("sde", 0),
                classifier_prob=cand.get("confidence", 0),
                classifier_class=cand.get("classification", "OTHER")
            )

            result = {
                **cand,
                **params.to_dict(),
                **metrics.to_dict(),
            }
            fitted_results.append(result)

            logger.info(
                f"TIC {cand.get('tic_id', '?')}: "
                f"P={params.period:.4f}d, "
                f"depth={params.depth_ppm:.0f}ppm, "
                f"dur={params.duration_hours:.2f}h, "
                f"SNR={metrics.transit_snr:.1f}, "
                f"class={cand.get('classification', '?')}"
            )

        except Exception as e:
            logger.warning(f"Fitting failed for TIC {cand.get('tic_id', '?')}: {e}")

        tracker.update()

    # Save results
    if fitted_results:
        results_df = pd.DataFrame([{k: v for k, v in r.items()
                                      if not isinstance(v, (np.ndarray, dict))}
                                     for r in fitted_results])
        results_df.to_csv(config.RESULTS_DIR / "fitted_results.csv", index=False)
        save_pickle(fitted_results, config.RESULTS_DIR / "fitted_results.pkl")
        save_json({str(i): {k: v for k, v in r.items()
                             if not isinstance(v, np.ndarray)}
                    for i, r in enumerate(fitted_results)},
                   config.RESULTS_DIR / "fitted_results.json")

        logger.info(f"Fitting complete: {len(fitted_results)} candidates fitted")

        # Identify strongest planet candidates
        planets = [r for r in fitted_results
                   if r.get("classification") == "PLANET"
                   and r.get("combined_confidence", 0) > 0.5]
        logger.info(f"Strong planet candidates: {len(planets)}")
    else:
        logger.warning("No transit fits completed")

    return fitted_results


@timer
def stage_visualize(fitted_results: List[Dict] = None):
    """Stage 7: Generate visualization plots."""
    from src.visualization import (plot_candidate_summary, plot_phase_folded,
                                    plot_periodogram, plot_raw_lightcurve)

    logger.info(f"=== STAGE 7: VISUALIZATION ===")

    # Load results
    if fitted_results is None:
        res_path = config.RESULTS_DIR / "fitted_results.pkl"
        if res_path.exists():
            fitted_results = load_pickle(res_path)
        else:
            logger.error("No fitted results found. Run fit stage first.")
            return

    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    tracker = ProgressTracker(len(fitted_results), "Generating plots")

    for i, result in enumerate(fitted_results):
        try:
            filepath = result.get("filepath")
            if not filepath or not Path(filepath).exists():
                tracker.update()
                continue

            data = load_pickle(filepath)
            t = data.get("time", np.array([]))
            f = data.get("flux", np.array([]))

            tic_id = str(result.get("tic_id", f"unknown_{i}"))
            classification = result.get("classification", "OTHER")

            # Multi-panel summary
            plot_candidate_summary(
                time=t, flux=f,
                period=result.get("period", 1),
                epoch=result.get("epoch", 0),
                model_flux=result.get("model_flux"),
                duration=result.get("duration"),
                class_probs=result.get("class_probs"),
                params=result,
                confidence=result,
                sde=result.get("sde", result.get("bls_sde")),
                tic_id=tic_id,
                classification=classification,
                save_path=config.PLOTS_DIR / f"summary_TIC{tic_id}.png"
            )

        except Exception as e:
            logger.warning(f"Plot generation failed for result {i}: {e}")

        tracker.update()

    logger.info(f"Visualization complete. Plots saved to {config.PLOTS_DIR}")


@timer
def stage_report(fitted_results: List[Dict] = None):
    """Stage 8: Generate PDF report."""
    from src.report_generator import generate_report

    logger.info(f"=== STAGE 8: REPORT GENERATION ===")

    if fitted_results is None:
        res_path = config.RESULTS_DIR / "fitted_results.pkl"
        if res_path.exists():
            fitted_results = load_pickle(res_path)

    generate_report(fitted_results)
    logger.info(f"Report saved to {config.REPORT_DIR}")



@timer
def run_all(sector: int = None, sample: int = None):
    """Execute the complete pipeline end-to-end."""
    logger.info("===================================================================╗")
    logger.info("║  EXOPLANET DETECTION PIPELINE — FULL RUN            ║")
    logger.info("===================================================================╝")

    start = _time.time()

    # 1. Download data
    tic_ids = stage_download(sector=sector, sample=sample)

    # 2. Preprocess
    stage_preprocess()

    # 3. Detect signals
    detections = stage_detect()

    # 4. Extract features
    valid_detections = stage_extract_features(detections)

    # 5. Classify
    classifications = stage_classify(valid_detections)

    # 6. Fit transit models
    fitted_results = stage_fit(classifications)

    # 7. Generate visualizations
    stage_visualize(fitted_results)

    # 8. Generate report
    stage_report(fitted_results)

    elapsed = _time.time() - start
    logger.info(f"Pipeline complete in {elapsed / 60:.1f} minutes")

    return fitted_results



def main():
    parser = argparse.ArgumentParser(
        description="AI-Enabled Exoplanet Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py run-all --sector 1 --sample 100
  python pipeline.py download --sector 1 --sample 50
  python pipeline.py preprocess
  python pipeline.py detect
  python pipeline.py classify
  python pipeline.py fit
  python pipeline.py visualize
  python pipeline.py report
        """
    )

    parser.add_argument("stage", choices=[
        "download", "preprocess", "detect", "extract",
        "classify", "fit", "visualize", "report", "run-all"
    ], help="Pipeline stage to execute")

    parser.add_argument("--sector", type=int, default=config.TESS_SECTOR,
                        help=f"TESS sector number (default: {config.TESS_SECTOR})")
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of light curves to download (default: all)")
    parser.add_argument("--tic-ids", type=str, default=None,
                        help="Comma-separated TIC IDs to analyze")

    args = parser.parse_args()

    logger.info(f"Pipeline stage: {args.stage}")

    if args.stage == "download":
        stage_download(sector=args.sector, sample=args.sample)
    elif args.stage == "preprocess":
        stage_preprocess()
    elif args.stage == "detect":
        stage_detect()
    elif args.stage == "extract":
        detections = stage_detect()
        stage_extract_features(detections)
    elif args.stage == "classify":
        stage_classify()
    elif args.stage == "fit":
        stage_fit()
    elif args.stage == "visualize":
        stage_visualize()
    elif args.stage == "report":
        stage_report()
    elif args.stage == "run-all":
        run_all(sector=args.sector, sample=args.sample)


if __name__ == "__main__":
    main()
