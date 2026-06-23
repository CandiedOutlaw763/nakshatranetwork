"""
ML Classification Module for the Exoplanet Detection Pipeline.

Implements a dual-branch CNN (AstroNet-inspired architecture) and Random Forest
classifier with ensemble prediction for classifying TESS transit signals into:
PLANET, ECLIPSING_BINARY, BLEND, or OTHER.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.utils import logger, timer

# TensorFlow imports (deferred logging level to reduce noise)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, Model

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)



def build_cnn_model() -> Model:
    """Build and compile the dual-branch CNN for transit classification.

    Architecture:
        - Global branch: processes full-orbit phase-folded light curve (201 bins)
        - Local branch: processes zoomed transit view (61 bins)
        - Both branches are concatenated and passed through a dense head
          with softmax output over 4 classes.

    Returns:
        Compiled Keras Model ready for training.
    """
    # --- Global branch (full orbital phase) ---
    global_input = layers.Input(
        shape=(config.GLOBAL_VIEW_BINS, 1), name="global_input"
    )
    x = layers.Conv1D(
        config.CNN_GLOBAL_FILTERS[0], kernel_size=5, padding="same",
        activation="relu", name="global_conv1"
    )(global_input)
    x = layers.Conv1D(
        config.CNN_GLOBAL_FILTERS[1], kernel_size=5, padding="same",
        activation="relu", name="global_conv2"
    )(x)
    x = layers.MaxPooling1D(pool_size=2, name="global_pool")(x)
    x = layers.Conv1D(
        config.CNN_GLOBAL_FILTERS[2], kernel_size=3, padding="same",
        activation="relu", name="global_conv3"
    )(x)
    global_out = layers.GlobalAveragePooling1D(name="global_gap")(x)

    # --- Local branch (zoomed transit window) ---
    local_input = layers.Input(
        shape=(config.LOCAL_VIEW_BINS, 1), name="local_input"
    )
    y = layers.Conv1D(
        config.CNN_LOCAL_FILTERS[0], kernel_size=5, padding="same",
        activation="relu", name="local_conv1"
    )(local_input)
    y = layers.Conv1D(
        config.CNN_LOCAL_FILTERS[1], kernel_size=3, padding="same",
        activation="relu", name="local_conv2"
    )(y)
    y = layers.MaxPooling1D(pool_size=2, name="local_pool")(y)
    y = layers.Conv1D(
        config.CNN_LOCAL_FILTERS[2], kernel_size=3, padding="same",
        activation="relu", name="local_conv3"
    )(y)
    local_out = layers.GlobalAveragePooling1D(name="local_gap")(y)

    # --- Merge branches ---
    merged = layers.Concatenate(name="merge")([global_out, local_out])

    # --- Classification head ---
    z = layers.Dense(
        config.CNN_DENSE_UNITS[0], activation="relu", name="dense1"
    )(merged)
    z = layers.Dropout(config.CNN_DROPOUT_RATE, name="dropout")(z)
    z = layers.Dense(
        config.CNN_DENSE_UNITS[1], activation="relu", name="dense2"
    )(z)
    output = layers.Dense(
        config.N_CLASSES, activation="softmax", name="output"
    )(z)

    model = Model(
        inputs=[global_input, local_input],
        outputs=output,
        name="AstroNet_Classifier",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.CNN_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    logger.info(
        f"Built CNN model: {model.count_params():,} parameters "
        f"({config.N_CLASSES} classes)"
    )
    return model


@timer
def train_cnn(
    model: Model,
    X_global: np.ndarray,
    X_local: np.ndarray,
    y: np.ndarray,
    val_data: Optional[Tuple[list, np.ndarray]] = None,
) -> keras.callbacks.History:
    """Train the CNN model with training callbacks.

    Args:
        model: Compiled Keras model from ``build_cnn_model``.
        X_global: Global-view inputs of shape ``(N, 201, 1)``.
        X_local: Local-view inputs of shape ``(N, 61, 1)``.
        y: One-hot encoded labels of shape ``(N, 4)``.
        val_data: Optional tuple ``([X_global_val, X_local_val], y_val)``.

    Returns:
        Keras History object containing training metrics per epoch.
    """
    checkpoint_path = config.MODELS_DIR / "cnn_best.keras"

    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss" if val_data else "loss",
            patience=config.CNN_EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss" if val_data else "loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss" if val_data else "loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    logger.info(
        f"Training CNN: {len(X_global)} samples, "
        f"epochs={config.CNN_EPOCHS}, batch_size={config.CNN_BATCH_SIZE}"
    )

    history = model.fit(
        [X_global, X_local],
        y,
        epochs=config.CNN_EPOCHS,
        batch_size=config.CNN_BATCH_SIZE,
        validation_data=val_data,
        callbacks=cb_list,
        verbose=1,
    )

    if history.history.get("loss"):
        logger.info(
            f"CNN training complete -- final loss: {history.history['loss'][-1]:.4f}"
        )
    else:
        logger.warning("CNN training produced no loss history (0 training samples?).")
    return history


def predict_cnn(
    model: Model, X_global: np.ndarray, X_local: np.ndarray
) -> np.ndarray:
    """Generate class probabilities from the CNN model.

    Args:
        model: Trained Keras model.
        X_global: Global-view inputs ``(N, 201, 1)``.
        X_local: Local-view inputs ``(N, 61, 1)``.

    Returns:
        Class probability array of shape ``(N, 4)``.
    """
    probs = model.predict([X_global, X_local], verbose=0)
    logger.debug(f"CNN predictions generated for {len(X_global)} samples")
    return probs



def build_rf_model() -> RandomForestClassifier:
    """Build a configured Random Forest classifier.

    Returns:
        Unfitted ``RandomForestClassifier`` with pipeline-tuned hyperparameters.
    """
    rf = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
        class_weight=config.RF_CLASS_WEIGHT,
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    logger.info(
        f"Built RF model: n_estimators={config.RF_N_ESTIMATORS}, "
        f"max_depth={config.RF_MAX_DEPTH}"
    )
    return rf


@timer
def train_rf(
    model: RandomForestClassifier, X_features: np.ndarray, y: np.ndarray
) -> RandomForestClassifier:
    """Train the Random Forest classifier.

    Args:
        model: Unfitted ``RandomForestClassifier``.
        X_features: Feature matrix ``(N, n_features)``.
        y: Integer class labels ``(N,)`` (not one-hot).

    Returns:
        Trained ``RandomForestClassifier``.
    """
    logger.info(
        f"Training RF: {X_features.shape[0]} samples, "
        f"{X_features.shape[1]} features"
    )
    model.fit(X_features, y)
    train_acc = model.score(X_features, y)
    logger.info(f"RF training complete -- train accuracy: {train_acc:.4f}")
    return model


def predict_rf(
    model: RandomForestClassifier, X_features: np.ndarray
) -> np.ndarray:
    """Generate class probabilities from the Random Forest model.

    Args:
        model: Trained ``RandomForestClassifier``.
        X_features: Feature matrix ``(N, n_features)``.

    Returns:
        Class probability array of shape ``(N, 4)``.
    """
    raw_probs = model.predict_proba(X_features)
    # RF may have fewer classes than config.N_CLASSES if some classes
    # were absent from training data. Pad to full width.
    if raw_probs.shape[1] < config.N_CLASSES:
        probs = np.zeros((raw_probs.shape[0], config.N_CLASSES), dtype=raw_probs.dtype)
        for i, cls in enumerate(model.classes_):
            probs[:, int(cls)] = raw_probs[:, i]
    else:
        probs = raw_probs
    logger.debug(f"RF predictions generated for {len(X_features)} samples")
    return probs



def ensemble_predict(
    cnn_probs: np.ndarray,
    rf_probs: np.ndarray,
    cnn_weight: float = config.CNN_WEIGHT,
    rf_weight: float = config.RF_WEIGHT,
) -> np.ndarray:
    """Combine CNN and RF predictions via weighted averaging.

    Args:
        cnn_probs: CNN class probabilities ``(N, 4)``.
        rf_probs: RF class probabilities ``(N, 4)``.
        cnn_weight: Weight applied to CNN predictions (default from config).
        rf_weight: Weight applied to RF predictions (default from config).

    Returns:
        Ensemble class probabilities ``(N, 4)`` (normalized to sum to 1).
    """
    total_weight = cnn_weight + rf_weight
    combined = (cnn_weight * cnn_probs + rf_weight * rf_probs) / total_weight
    logger.debug(
        f"Ensemble prediction: CNN weight={cnn_weight}, RF weight={rf_weight}"
    )
    return combined


def classify(
    global_view: np.ndarray,
    local_view: np.ndarray,
    features: np.ndarray,
    cnn_model: Model,
    rf_model: RandomForestClassifier,
) -> Dict[str, Any]:
    """Run full ensemble classification on a batch of candidates.

    Args:
        global_view: Global phase-folded views ``(N, 201, 1)``.
        local_view: Local phase-folded views ``(N, 61, 1)``.
        features: Extracted feature vectors ``(N, n_features)``.
        cnn_model: Trained CNN model.
        rf_model: Trained Random Forest model.

    Returns:
        Dictionary with keys:
            - ``predicted_classes``: list of class label strings
            - ``class_indices``: integer class indices ``(N,)``
            - ``probabilities``: ensemble probabilities ``(N, 4)``
            - ``confidence``: max probability per sample ``(N,)``
            - ``cnn_probs``: raw CNN probabilities ``(N, 4)``
            - ``rf_probs``: raw RF probabilities ``(N, 4)``
    """
    cnn_probs = predict_cnn(cnn_model, global_view, local_view)
    rf_probs = predict_rf(rf_model, features)
    ensemble_probs = ensemble_predict(cnn_probs, rf_probs)

    class_indices = np.argmax(ensemble_probs, axis=1)
    predicted_classes = [
        config.CLASSIFICATION_CLASSES[idx] for idx in class_indices
    ]
    confidence = np.max(ensemble_probs, axis=1)

    logger.info(
        f"Classification complete for {len(global_view)} candidates — "
        f"mean confidence: {confidence.mean():.3f}"
    )

    return {
        "predicted_classes": predicted_classes,
        "class_indices": class_indices,
        "probabilities": ensemble_probs,
        "confidence": confidence,
        "cnn_probs": cnn_probs,
        "rf_probs": rf_probs,
    }



def save_models(cnn_model: Model, rf_model: RandomForestClassifier,
                path: Path = None) -> None:
    """Save both CNN and RF models to disk.

    Args:
        cnn_model: Trained Keras CNN model.
        rf_model: Trained scikit-learn RF model.
        path: Directory to save models in. Defaults to ``config.MODELS_DIR``.
    """
    path = Path(path) if path else config.MODELS_DIR
    path.mkdir(parents=True, exist_ok=True)

    cnn_path = path / "cnn_model.keras"
    rf_path = path / "rf_model.joblib"

    cnn_model.save(str(cnn_path))
    joblib.dump(rf_model, rf_path)

    logger.info(f"Saved CNN model  -> {cnn_path}")
    logger.info(f"Saved RF model   -> {rf_path}")


def load_models(path: Path = None) -> Tuple[Model, RandomForestClassifier]:
    """Load both CNN and RF models from disk.

    Args:
        path: Directory containing saved models.
              Defaults to ``config.MODELS_DIR``.

    Returns:
        Tuple of ``(cnn_model, rf_model)``.

    Raises:
        FileNotFoundError: If either model file is missing.
    """
    path = Path(path) if path else config.MODELS_DIR

    cnn_path = path / "cnn_model.keras"
    rf_path = path / "rf_model.joblib"

    if not cnn_path.exists():
        raise FileNotFoundError(f"CNN model not found: {cnn_path}")
    if not rf_path.exists():
        raise FileNotFoundError(f"RF model not found: {rf_path}")

    cnn_model = keras.models.load_model(str(cnn_path))
    rf_model = joblib.load(rf_path)

    logger.info(f"Loaded CNN model <- {cnn_path}")
    logger.info(f"Loaded RF model  <- {rf_path}")
    return cnn_model, rf_model



def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate classification performance with multiple metrics.

    Args:
        y_true: True integer class labels ``(N,)``.
        y_pred: Predicted integer class labels ``(N,)``.
        y_probs: Predicted class probabilities ``(N, 4)``.

    Returns:
        Dictionary containing:
            - ``classification_report``: per-class precision / recall / F1
            - ``confusion_matrix``: ``(4, 4)`` confusion matrix
            - ``roc_auc``: macro-averaged ROC-AUC (one-vs-rest)
            - ``accuracy``: overall accuracy
    """
    # Classification report (dict form for JSON serialization)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(config.N_CLASSES)),
        target_names=config.CLASSIFICATION_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(config.N_CLASSES))
    )

    # ROC-AUC (one-vs-rest, macro)
    try:
        # One-hot encode y_true for ROC-AUC computation
        y_true_onehot = np.eye(config.N_CLASSES)[y_true.astype(int)]
        roc_auc = roc_auc_score(
            y_true_onehot, y_probs, multi_class="ovr", average="macro"
        )
    except ValueError as e:
        logger.warning(f"ROC-AUC computation failed: {e}")
        roc_auc = None

    accuracy = float(np.mean(y_true == y_pred))

    # Log summary
    logger.info("=" * 60)
    logger.info("CLASSIFICATION EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Accuracy : {accuracy:.4f}")
    if roc_auc is not None:
        logger.info(f"ROC-AUC  : {roc_auc:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(config.N_CLASSES)),
            target_names=config.CLASSIFICATION_CLASSES,
            zero_division=0,
        )
    )

    return {
        "classification_report": report,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "accuracy": accuracy,
    }
