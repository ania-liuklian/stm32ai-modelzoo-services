"""Fine-tune ST-IGN for six-axis IMU CSV data and report held-out metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from human_activity_recognition.tf.src.datasets.realworld_imu import prepare_realworld_imu_arrays
from human_activity_recognition.tf.src.models.ign import get_ign
from human_activity_recognition.tf.src.models.imu_transfer import transfer_st_ign_weights


def _metrics(model, x, y, class_names, output_dir, prefix, model_classes=None):
    probabilities = model.predict(x[..., np.newaxis], verbose=0)
    model_classes = model_classes or class_names
    native_predictions = probabilities.argmax(axis=1)
    predicted_names = [model_classes[i] for i in native_predictions]
    predicted = np.asarray([class_names.index(name) for name in predicted_names])
    report = classification_report(
        y, predicted, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, predicted, labels=list(range(len(class_names))))
    np.savetxt(output_dir / f"{prefix}_confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "per_class": report,
        "confusion_matrix": cm.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Merged synchronized six-axis CSV")
    parser.add_argument("--source-model", help="Existing three-axis .keras/.h5 checkpoint")
    parser.add_argument("--classes", required=True, help="Comma-separated target labels in output order")
    parser.add_argument("--source-classes", default="Stationary,Walking,Jogging,Biking",
                        help="Output order used to train the source checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--scratch", action="store_true", help="Train six-axis baseline from random initialization")
    parser.add_argument("--baseline-preprocessing-matches", action="store_true",
                        help="Assert source checkpoint preprocessing matches the six-axis test data")
    args = parser.parse_args()
    if not args.scratch and not args.source_model:
        parser.error("--source-model is required unless --scratch is selected")

    target_classes = [name.strip() for name in args.classes.split(",") if name.strip()]
    source_classes = [name.strip() for name in args.source_classes.split(",") if name.strip()]
    if len(target_classes) < 2 or len(set(target_classes)) != len(target_classes):
        parser.error("--classes must contain at least two unique labels")
    tf.keras.utils.set_random_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_realworld_imu_arrays(
        args.csv, target_classes, args.sequence_length,
        args.validation_split, args.test_split, args.seed, normalization=True)
    x_train, y_train = data["x_train"][..., np.newaxis], data["y_train"]
    x_valid, y_valid = data["x_valid"][..., np.newaxis], data["y_valid"]
    x_test, y_test = data["x_test"][..., np.newaxis], data["y_test"]
    model = get_ign((args.sequence_length, 6, 1), len(target_classes), dropout=0.5)

    transfer_report = None
    source = None
    if not args.scratch:
        source = tf.keras.models.load_model(args.source_model, compile=False)
        transfer_report = transfer_st_ign_weights(
            source, model, source_classes, target_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            output_dir / "best_six_axis.keras", monitor="val_loss",
            save_best_only=True, mode="min"),
        tf.keras.callbacks.ModelCheckpoint(
            output_dir / "last_six_axis.keras", save_best_only=False),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(output_dir / "training_metrics.csv"),
    ]
    model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
              batch_size=args.batch_size, epochs=args.epochs, callbacks=callbacks,
              shuffle=True, verbose=2)
    model = tf.keras.models.load_model(output_dir / "best_six_axis.keras", compile=False)
    results = {
        "target_classes": target_classes,
        "source_classes": source_classes,
        "sequence_length": args.sequence_length,
        "input_layout": "(batch, time, axes, 1), channels_last",
        "transfer": transfer_report,
        "six_axis": _metrics(model, x_test[..., 0], y_test, target_classes, output_dir, "six_axis"),
    }
    # An accelerometer-only comparison is meaningful only when source and target
    # class meanings match and the source preprocessing is also reproduced.
    if (source is not None and set(source_classes) == set(target_classes) and
            args.baseline_preprocessing_matches):
        baseline = _metrics(source, x_test[..., :3, 0], y_test, target_classes,
                            output_dir, "accelerometer_baseline",
                            model_classes=source_classes)
        results["accelerometer_baseline"] = baseline
    else:
        results["accelerometer_baseline"] = {
            "not_run": ("Source and target class mappings differ, source preprocessing is unverified, "
                        "or no source checkpoint was loaded; a baseline score would be misleading.")
        }
    with (output_dir / "evaluation_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with (output_dir / "normalization.json").open("w", encoding="utf-8") as f:
        json.dump({"channels": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
                   "mean": data["mean"].tolist(), "std": data["std"].tolist()}, f, indent=2)
    print(json.dumps({k: v for k, v in results.items()
                      if k in ("transfer", "six_axis", "accelerometer_baseline")}, indent=2))


if __name__ == "__main__":
    main()
