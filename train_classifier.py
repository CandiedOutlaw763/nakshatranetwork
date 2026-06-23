#!/usr/bin/env python3
"""
Training script for the Exoplanet Classification Pipeline.

Loads preprocessed TESS light-curve data, trains CNN + Random Forest models,
evaluates ensemble performance on a held-out test set, and persists all
artefacts (models, metrics, plots) for downstream use.

Usage:
    python train_classifier.py
    python train_classifier.py --epochs 50 --batch-size 32 --sector 1
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import config
from src.utils import logger, timer, save_json, load_pickle
from src.classifier import (
    build_cnn_model,
    train_cnn,
    predict_cnn,
    build_rf_model,
    train_rf,
    predict_rf,
    ensemble_predict,
    evaluate,
    save_models,
)

# Attempt to import feature extraction (may not exist yet)
try:
    from src.feature_extraction import extract_features
except ImportError:
    extract_features = None
    logger.warning(
        "src.feature_extraction not available — "
        "will look for pre-computed feature files instead."
    )



@timer
def load_data(sector: int) -> Dict[str, np.ndarray]:
    """Load or generate preprocessed light-curve views, features, and labels."""
    processed_dir = config.PROCESSED_DATA_DIR
    catalog_dir = config.CATALOG_DIR
    results_dir = config.RESULTS_DIR

    global_path = processed_dir / f"global_views_sector{sector}.npy"
    local_path = processed_dir / f"local_views_sector{sector}.npy"
    features_path = processed_dir / f"features_sector{sector}.npy"
    labels_path = processed_dir / f"labels_sector{sector}.npy"

    if global_path.exists() and local_path.exists() and features_path.exists() and labels_path.exists():
        global_views = np.load(global_path)
        local_views = np.load(local_path)
        features = np.load(features_path)
        labels = np.load(labels_path)
        logger.info(f"Loaded existing training data for sector {sector}")
    else:
        logger.info("Training arrays not found. Building training dataset from pipeline outputs...")
        
        # Check prerequisites
        labeled_dataset_path = catalog_dir / "labeled_dataset.csv"
        if not labeled_dataset_path.exists():
            raise FileNotFoundError(
                f"Labeled dataset not found at {labeled_dataset_path}\n"
                "Please run `python pipeline.py download` to acquire data."
            )
            
        detections_path = results_dir / "detections.csv"
        if not detections_path.exists():
            raise FileNotFoundError(
                f"Detection results not found at {detections_path}\n"
                "Please run `python pipeline.py preprocess` followed by `python pipeline.py detect`."
            )
            
        from src.feature_extraction import prepare_training_data
        
        catalog_df = pd.read_csv(labeled_dataset_path)
        detection_results_df = pd.read_csv(detections_path)
        
        logger.info("Running feature extraction on detected signals...")
        global_views, local_views, features, labels = prepare_training_data(
            preprocessed_dir=processed_dir,
            catalog_df=catalog_df,
            detection_results_df=detection_results_df
        )
        
        # Save for future use
        np.save(global_path, global_views)
        np.save(local_path, local_views)
        np.save(features_path, features)
        np.save(labels_path, labels)
        logger.info(f"Saved generated training dataset to {processed_dir}")

    # Map labels back to names for reporting
    idx_to_class = {idx: name for idx, name in enumerate(config.CLASSIFICATION_CLASSES)}
    label_names = np.array([idx_to_class.get(l, "OTHER") for l in labels])

    logger.info(
        f"Dataset sizes - global: {global_views.shape}, local: {local_views.shape}, "
        f"features: {features.shape}, labels: {labels.shape}"
    )

    return {
        "global_views": global_views,
        "local_views": local_views,
        "features": features,
        "labels": labels,
        "label_names": label_names,
    }



def split_data(
    data: Dict[str, np.ndarray],
    train_frac: float = config.TRAIN_FRACTION,
    val_frac: float = config.VAL_FRACTION,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Stratified-random split into train / validation / test sets.

    Args:
        data: Dictionary returned by ``load_data``.
        train_frac: Fraction of samples for training.
        val_frac: Fraction of samples for validation.

    Returns:
        Three dictionaries (train, val, test) with the same keys as *data*.
    """
    n = len(data["labels"])
    indices = np.arange(n)
    rng = np.random.default_rng(seed=42)
    rng.shuffle(indices)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    def _subset(idx: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "global_views": data["global_views"][idx],
            "local_views": data["local_views"][idx],
            "features": data["features"][idx],
            "labels": data["labels"][idx],
            "label_names": data["label_names"][idx],
        }

    train, val, test = _subset(train_idx), _subset(val_idx), _subset(test_idx)
    logger.info(
        f"Split: train={len(train['labels'])}, "
        f"val={len(val['labels'])}, test={len(test['labels'])}"
    )
    return train, val, test



def _apply_plot_style() -> None:
    """Apply the dark-background matplotlib style from config."""
    try:
        plt.style.use(config.PLOT_STYLE)
    except OSError:
        plt.style.use("dark_background")


def plot_training_curves(
    history,
    save_path: Path = None,
) -> None:
    """Plot CNN training / validation loss and accuracy curves.

    Args:
        history: Keras ``History`` object.
        save_path: File path to save the figure. Defaults to
                   ``config.PLOTS_DIR / 'training_curves.png'``.
    """
    _apply_plot_style()
    save_path = save_path or config.PLOTS_DIR / "training_curves.png"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history.history["loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history.history["loss"], label="Train Loss",
                 color=config.COLORS["accent_blue"], linewidth=2)
    if "val_loss" in history.history:
        axes[0].plot(epochs, history.history["val_loss"], label="Val Loss",
                     color=config.COLORS["accent_red"], linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history.history["accuracy"], label="Train Accuracy",
                 color=config.COLORS["accent_green"], linewidth=2)
    if "val_accuracy" in history.history:
        axes[1].plot(epochs, history.history["val_accuracy"],
                     label="Val Accuracy",
                     color=config.COLORS["accent_yellow"], linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("CNN Training Curves", fontsize=14, fontweight="bold",
                 color=config.COLORS["text_primary"])
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Saved training curves -> {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list = None,
    save_path: Path = None,
) -> None:
    """Plot a styled confusion matrix heatmap.

    Args:
        cm: Confusion matrix array ``(N_CLASSES, N_CLASSES)``.
        class_names: Class label strings.
        save_path: File path to save the figure. Defaults to
                   ``config.PLOTS_DIR / 'confusion_matrix.png'``.
    """
    _apply_plot_style()
    class_names = class_names or config.CLASSIFICATION_CLASSES
    save_path = save_path or config.PLOTS_DIR / "confusion_matrix.png"

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="magma")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12, fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Saved confusion matrix -> {save_path}")



@timer
def run_training(
    sector: int,
    epochs: int,
    batch_size: int,
) -> Dict[str, Any]:
    """End-to-end training pipeline.

    1. Load preprocessed data & labels
    2. Split into train / val / test
    3. Train CNN and RF models
    4. Evaluate ensemble on the test set
    5. Save models, metrics, and plots

    Args:
        sector: TESS sector number.
        epochs: Number of CNN training epochs.
        batch_size: CNN mini-batch size.

    Returns:
        Evaluation results dictionary.
    """
    # Override config values from CLI args
    config.CNN_EPOCHS = epochs
    config.CNN_BATCH_SIZE = batch_size

    # ---- Load Data ----
    logger.info("=" * 60)
    logger.info(f"TRAINING CLASSIFIER -- Sector {sector}")
    logger.info("=" * 60)
    data = load_data(sector)

    # ---- Split ----
    train, val, test = split_data(data)

    # ---- Prepare CNN inputs (add channel dim if needed) ----
    def _ensure_3d(arr: np.ndarray) -> np.ndarray:
        return arr[..., np.newaxis] if arr.ndim == 2 else arr

    X_train_global = _ensure_3d(train["global_views"])
    X_train_local = _ensure_3d(train["local_views"])
    X_val_global = _ensure_3d(val["global_views"])
    X_val_local = _ensure_3d(val["local_views"])
    X_test_global = _ensure_3d(test["global_views"])
    X_test_local = _ensure_3d(test["local_views"])

    # One-hot encode labels for CNN
    from tensorflow.keras.utils import to_categorical

    y_train_cat = to_categorical(train["labels"], num_classes=config.N_CLASSES)
    y_val_cat = to_categorical(val["labels"], num_classes=config.N_CLASSES)

    # ---- Train CNN ----
    cnn_model = build_cnn_model()

    history = train_cnn(
        cnn_model,
        X_train_global,
        X_train_local,
        y_train_cat,
        val_data=([X_val_global, X_val_local], y_val_cat),
    )

    # ---- Train RF ----
    rf_model = build_rf_model()
    rf_model = train_rf(rf_model, train["features"], train["labels"])

    # ---- Evaluate on Test Set ----
    logger.info("Evaluating on test set...")
    cnn_probs = predict_cnn(cnn_model, X_test_global, X_test_local)
    rf_probs = predict_rf(rf_model, test["features"])
    ensemble_probs = ensemble_predict(cnn_probs, rf_probs)

    y_pred = np.argmax(ensemble_probs, axis=1)
    results = evaluate(test["labels"], y_pred, ensemble_probs)

    # ---- Save Models ----
    save_models(cnn_model, rf_model, config.MODELS_DIR)

    # ---- Save Classification Report ----
    report_path = config.RESULTS_DIR / "classification_report.json"
    save_json(
        {
            "sector": sector,
            "epochs_trained": len(history.history.get("loss", [])),
            "accuracy": results["accuracy"],
            "roc_auc": results["roc_auc"],
            "classification_report": results["classification_report"],
            "confusion_matrix": results["confusion_matrix"].tolist(),
        },
        report_path,
    )
    logger.info(f"Saved classification report -> {report_path}")

    # ---- Plots ----
    plot_training_curves(history)
    plot_confusion_matrix(results["confusion_matrix"])

    # ---- Summary ----
    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Sector          : {sector}")
    logger.info(f"  Train samples   : {len(train['labels'])}")
    logger.info(f"  Val samples     : {len(val['labels'])}")
    logger.info(f"  Test samples    : {len(test['labels'])}")
    logger.info(f"  CNN epochs run  : {len(history.history.get('loss', []))}")
    logger.info(f"  Test accuracy   : {results['accuracy']:.4f}")
    if results["roc_auc"] is not None:
        logger.info(f"  Test ROC-AUC    : {results['roc_auc']:.4f}")
    logger.info("=" * 60)

    return results



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the exoplanet transit classifier (CNN + RF ensemble).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.CNN_EPOCHS,
        help="Number of CNN training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.CNN_BATCH_SIZE,
        help="CNN mini-batch size.",
    )
    parser.add_argument(
        "--sector",
        type=int,
        default=config.TESS_SECTOR,
        help="TESS sector number to train on.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    logger.info(
        f"CLI args: epochs={args.epochs}, batch_size={args.batch_size}, "
        f"sector={args.sector}"
    )
    run_training(
        sector=args.sector,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
