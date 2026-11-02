"""CSV loader for synchronized RealWorld accelerometer + gyroscope samples."""
import numpy as np
import pandas as pd
import tensorflow as tf

from typing import List, Tuple
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras.utils import to_categorical


CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
ALIASES = {
    "acc_x": ("acc_x", "accel_x", "accelerometer_x", "x_acc", "accX"),
    "acc_y": ("acc_y", "accel_y", "accelerometer_y", "y_acc", "accY"),
    "acc_z": ("acc_z", "accel_z", "accelerometer_z", "z_acc", "accZ"),
    "gyro_x": ("gyro_x", "gyroscope_x", "gyr_x", "x_gyro", "gyroX"),
    "gyro_y": ("gyro_y", "gyroscope_y", "gyr_y", "y_gyro", "gyroY"),
    "gyro_z": ("gyro_z", "gyroscope_z", "gyr_z", "z_gyro", "gyroZ"),
}


def _column(df, aliases):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in aliases:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def _read_samples(path):
    df = pd.read_csv(path)
    cols = {key: _column(df, aliases) for key, aliases in ALIASES.items()}
    missing = [key for key, value in cols.items() if value is None]
    label = _column(df, ("activity", "activity_label", "label", "gt", "class"))
    if missing or label is None:
        raise ValueError(
            "RealWorld IMU CSV must contain synchronized acc_x/acc_y/acc_z and "
            "gyro_x/gyro_y/gyro_z columns plus an activity/label column. "
            f"Could not find channels {missing}; label column found: {label is not None}."
        )
    # Preserve contiguous recording and subject boundaries where supplied.
    subject = _column(df, ("subject", "user", "participant", "person"))
    recording = _column(df, ("recording", "recording_id", "trial", "file_id"))
    df = df[[cols[c] for c in CHANNELS] + [label] +
            ([subject] if subject else []) + ([recording] if recording else [])].copy()
    df.columns = CHANNELS + ["label"] + (["subject"] if subject else []) + (["recording"] if recording else [])
    df = df.dropna(subset=CHANNELS + ["label"]).reset_index(drop=True)
    for c in CHANNELS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=CHANNELS).reset_index(drop=True)


def _windows(df, seq_len, class_names):
    group_cols = [c for c in ("subject", "recording") if c in df.columns]
    # Do not join adjacent trials or activity transitions into one window.
    groups = df.groupby(group_cols, sort=False, dropna=False) if group_cols else [(None, df)]
    xs, ys, groups_out = [], [], []
    for group_key, part in groups:
        part = part.reset_index(drop=True)
        # Each contiguous activity run is treated as a separate recording.
        run = part["label"].ne(part["label"].shift()).cumsum()
        for _, activity in part.groupby(run, sort=False):
            label = str(activity["label"].iloc[0])
            if label not in class_names:
                continue
            values = activity[CHANNELS].to_numpy(dtype=np.float32)
            n = len(values) // seq_len
            for i in range(n):
                xs.append(values[i * seq_len:(i + 1) * seq_len])
                ys.append(class_names.index(label))
                split_key = group_key[0] if isinstance(group_key, tuple) and "subject" in group_cols else group_key
                groups_out.append(str(split_key))
    if not xs:
        raise ValueError("No full labeled windows found. Check labels, classes and input_shape[0].")
    return np.asarray(xs), np.asarray(ys), np.asarray(groups_out)


def prepare_realworld_imu_arrays(training_path: str, class_names: List[str],
                                 sequence_length: int, validation_split: float,
                                 test_split: float, seed: int,
                                 normalization: bool = True):
    """Return leakage-safe splits and train-only channel normalization values."""
    df = _read_samples(training_path)
    split_group = "subject" if "subject" in df.columns else ("recording" if "recording" in df.columns else None)
    if split_group is None:
        raise ValueError("Provide a subject or recording column to prevent train/test leakage.")
    x, y, groups = _windows(df, sequence_length, class_names)
    # GroupShuffleSplit keeps all windows from a subject/recording in one partition.
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise ValueError("At least three distinct subjects/recordings are required for train/validation/test splits.")
    test_fraction = test_split
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    trainval_idx, test_idx = next(splitter.split(x, y, groups))
    val_fraction = validation_split
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_rel, val_rel = next(splitter.split(x[trainval_idx], y[trainval_idx], groups[trainval_idx]))
    train_idx, val_idx = trainval_idx[train_rel], trainval_idx[val_rel]
    mean = np.zeros((1, 1, len(CHANNELS)), dtype=np.float32)
    std = np.ones_like(mean)
    if normalization:
        mean = x[train_idx].mean(axis=(0, 1), keepdims=True)
        std = x[train_idx].std(axis=(0, 1), keepdims=True)
        std[std < 1e-6] = 1.0
        x = (x - mean) / std
    return {
        "x_train": x[train_idx], "y_train": y[train_idx],
        "x_valid": x[val_idx], "y_valid": y[val_idx],
        "x_test": x[test_idx], "y_test": y[test_idx],
        "mean": mean.reshape(-1), "std": std.reshape(-1),
        "class_names": list(class_names),
    }


def load_realworld_imu(training_path: str, class_names: List[str],
                       input_shape: Tuple[int], validation_split: float,
                       test_split: float, batch_size: int, seed: int,
                       normalization: bool = True, to_cache: bool = False, **kwargs):
    """Load one merged CSV; Keras input layout is (batch, time, axes, 1)."""
    if not training_path:
        return None, None, None
    if len(input_shape) != 3 or tuple(input_shape[1:]) != (6, 1):
        raise ValueError(f"RealWorld IMU expects (timesteps, 6, 1), got {input_shape}")
    data = prepare_realworld_imu_arrays(
        training_path, class_names, input_shape[0], validation_split,
        test_split, seed, normalization)

    def make_ds(split, shuffle):
        x = data[f"x_{split}"][..., np.newaxis]
        y = to_categorical(data[f"y_{split}"], num_classes=len(class_names))
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            ds = ds.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size)
        return ds.cache() if to_cache else ds
    return make_ds("train", True), make_ds("valid", True), make_ds("test", False)
