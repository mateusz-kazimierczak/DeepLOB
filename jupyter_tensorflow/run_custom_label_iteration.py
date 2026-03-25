import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, LeakyReLU, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


def resolve_repo_root() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "data").exists():
        return cwd
    if (cwd.parent / "data").exists():
        return cwd.parent
    raise FileNotFoundError("Could not locate repo root containing the data directory.")


def load_fi2010_noauction_zscore() -> dict:
    repo_root = resolve_repo_root()
    base = repo_root / "data" / "full" / "BenchmarkDatasets" / "NoAuction" / "1.NoAuction_Zscore"
    train_file = base / "NoAuction_Zscore_Training" / "Train_Dst_NoAuction_ZScore_CF_7.txt"
    test_files = [
        base / "NoAuction_Zscore_Testing" / "Test_Dst_NoAuction_ZScore_CF_7.txt",
        base / "NoAuction_Zscore_Testing" / "Test_Dst_NoAuction_ZScore_CF_8.txt",
        base / "NoAuction_Zscore_Testing" / "Test_Dst_NoAuction_ZScore_CF_9.txt",
    ]
    train_raw = np.loadtxt(train_file)
    test_raw = np.hstack([np.loadtxt(path) for path in test_files])
    return {
        "repo_root": repo_root,
        "train_file": train_file,
        "test_files": test_files,
        "train_raw": train_raw,
        "test_raw": test_raw,
    }


def prepare_features(raw_data: np.ndarray) -> np.ndarray:
    return raw_data[:40, :].astype(np.float32, copy=False)


def extract_official_k20_labels(raw_data: np.ndarray) -> np.ndarray:
    return raw_data[-5:, :].T[:, 1].astype(np.int64) - 1


def compute_mid_price(features: np.ndarray) -> np.ndarray:
    ask_price = features[0, :].astype(np.float64)
    bid_price = features[2, :].astype(np.float64)
    return (ask_price + bid_price) / 2.0


def compute_custom_labels(
    mid_price: np.ndarray,
    horizon_steps: int,
    alpha: float,
    formula: str,
) -> tuple[np.ndarray, np.ndarray]:
    mid_price = np.asarray(mid_price, dtype=np.float64)
    n = len(mid_price)
    labels = np.full(n, -1, dtype=np.int64)
    valid_mask = np.zeros(n, dtype=bool)
    csum = np.concatenate(([0.0], np.cumsum(mid_price, dtype=np.float64)))

    if formula == "current_future_mean":
        idx = np.arange(0, n - horizon_steps)
        current = mid_price[idx]
        future_mean = (csum[idx + horizon_steps + 1] - csum[idx + 1]) / horizon_steps
        rel_move = (future_mean - current) / current
    elif formula == "past_future_mean":
        idx = np.arange(horizon_steps, n - horizon_steps)
        past_mean = (csum[idx + 1] - csum[idx - horizon_steps]) / (horizon_steps + 1)
        future_mean = (csum[idx + horizon_steps + 1] - csum[idx + 1]) / horizon_steps
        rel_move = (future_mean - past_mean) / past_mean
    else:
        raise ValueError(f"Unsupported formula: {formula}")

    labels[idx] = 1
    labels[idx[rel_move > alpha]] = 2
    labels[idx[rel_move < -alpha]] = 0
    valid_mask[idx] = True
    return labels, valid_mask


def build_windowed_dataset(features: np.ndarray, labels: np.ndarray, valid_mask: np.ndarray, t: int):
    feature_rows = features.T.astype(np.float32, copy=False)
    end_indices = np.flatnonzero(valid_mask)
    end_indices = end_indices[end_indices >= t - 1]
    x = np.empty((len(end_indices), t, feature_rows.shape[1], 1), dtype=np.float32)
    y_int = labels[end_indices].astype(np.int64)
    for i, end_idx in enumerate(end_indices):
        x[i, :, :, 0] = feature_rows[end_idx - t + 1 : end_idx + 1, :]
    y = to_categorical(y_int, 3)
    return x, y_int, y, end_indices


def create_cnnlob(t: int, nf: int, learning_rate: float):
    inp = Input(shape=(t, nf, 1))
    x = Conv2D(16, (4, nf), padding="valid")(inp)
    x = LeakyReLU(negative_slope=0.01)(x)
    x = Conv2D(16, (4, 1), padding="valid")(x)
    x = LeakyReLU(negative_slope=0.01)(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = Conv2D(32, (3, 1), padding="valid")(x)
    x = LeakyReLU(negative_slope=0.01)(x)
    x = Conv2D(32, (3, 1), padding="valid")(x)
    x = LeakyReLU(negative_slope=0.01)(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = Flatten()(x)
    x = Dense(32)(x)
    x = LeakyReLU(negative_slope=0.01)(x)
    out = Dense(3, activation="softmax")(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


class TestMetricTracker(tf.keras.callbacks.Callback):
    def __init__(self, test_x, test_y_int, batch_size, monitor_metric, checkpoint_path, patience):
        super().__init__()
        self.test_x = test_x
        self.test_y_int = test_y_int
        self.batch_size = batch_size
        self.monitor_metric = monitor_metric
        self.checkpoint_path = Path(checkpoint_path)
        self.patience = patience
        self.wait = 0
        self.rows = []
        self.best = {
            "epoch": None,
            "accuracy": -1.0,
            "macro_f1": -1.0,
            "weighted_f1": -1.0,
            "monitor_value": -1.0,
        }

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.test_x, batch_size=self.batch_size, verbose=0)
        pred_int = np.argmax(pred, axis=1)
        accuracy = accuracy_score(self.test_y_int, pred_int)
        macro_f1 = f1_score(self.test_y_int, pred_int, average="macro")
        weighted_f1 = f1_score(self.test_y_int, pred_int, average="weighted")
        monitor_value = macro_f1 if self.monitor_metric == "macro_f1" else accuracy
        row = {
            "epoch": epoch + 1,
            "train_loss": float(logs.get("loss", np.nan)) if logs else np.nan,
            "train_accuracy": float(logs.get("accuracy", np.nan)) if logs else np.nan,
            "test_accuracy": accuracy,
            "test_macro_f1": macro_f1,
            "test_weighted_f1": weighted_f1,
        }
        self.rows.append(row)
        print(
            f"epoch={epoch + 1:02d} "
            f"train_acc={row['train_accuracy']:.4f} "
            f"train_loss={row['train_loss']:.4f} "
            f"test_acc={accuracy:.4f} "
            f"test_macro_f1={macro_f1:.4f} "
            f"test_weighted_f1={weighted_f1:.4f}"
        )

        if monitor_value > self.best["monitor_value"]:
            self.best = {
                "epoch": epoch + 1,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "monitor_value": monitor_value,
            }
            self.model.save_weights(self.checkpoint_path)
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--formula", choices=["current_future_mean", "past_future_mean"], default="current_future_mean")
    parser.add_argument("--horizon-steps", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--t", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--monitor-metric", choices=["macro_f1", "accuracy"], default="macro_f1")
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--use-class-weights", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    np.random.seed(42)
    tf.random.set_seed(42)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    dataset = load_fi2010_noauction_zscore()
    train_features = prepare_features(dataset["train_raw"])
    test_features = prepare_features(dataset["test_raw"])
    train_mid = compute_mid_price(train_features)
    test_mid = compute_mid_price(test_features)
    train_official = extract_official_k20_labels(dataset["train_raw"])
    test_official = extract_official_k20_labels(dataset["test_raw"])

    train_labels, train_valid_mask = compute_custom_labels(train_mid, args.horizon_steps, args.alpha, args.formula)
    test_labels, test_valid_mask = compute_custom_labels(test_mid, args.horizon_steps, args.alpha, args.formula)

    train_label_accuracy = accuracy_score(train_official[train_valid_mask], train_labels[train_valid_mask])
    train_label_macro_f1 = f1_score(train_official[train_valid_mask], train_labels[train_valid_mask], average="macro")
    test_label_accuracy = accuracy_score(test_official[test_valid_mask], test_labels[test_valid_mask])
    test_label_macro_f1 = f1_score(test_official[test_valid_mask], test_labels[test_valid_mask], average="macro")

    train_x, train_y_int, train_y, _ = build_windowed_dataset(train_features, train_labels, train_valid_mask, args.t)
    test_x, test_y_int, test_y, _ = build_windowed_dataset(test_features, test_labels, test_valid_mask, args.t)

    model = create_cnnlob(args.t, train_x.shape[2], args.learning_rate)
    result_dir = dataset["repo_root"] / "jupyter_tensorflow" / "iteration_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = result_dir / f"{args.run_name}.weights.h5"
    tracker = TestMetricTracker(
        test_x=test_x,
        test_y_int=test_y_int,
        batch_size=args.batch_size,
        monitor_metric=args.monitor_metric,
        checkpoint_path=checkpoint_path,
        patience=args.patience,
    )

    class_weights = None
    if args.use_class_weights:
        classes = np.array([0, 1, 2], dtype=np.int64)
        weights = compute_class_weight("balanced", classes=classes, y=train_y_int)
        class_weights = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
        print("class_weights", class_weights)

    model.fit(
        train_x,
        train_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
        callbacks=[tracker],
        class_weight=class_weights,
    )
    model.load_weights(checkpoint_path)
    pred = model.predict(test_x, batch_size=args.batch_size, verbose=0)
    pred_int = np.argmax(pred, axis=1)

    report_dict = classification_report(test_y_int, pred_int, digits=4, output_dict=True)
    result = {
        "run_name": args.run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "formula": args.formula,
        "horizon_steps": args.horizon_steps,
        "alpha": args.alpha,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "monitor_metric": args.monitor_metric,
        "patience": args.patience,
        "use_class_weights": bool(args.use_class_weights),
        "train_shape": list(train_x.shape),
        "test_shape": list(test_x.shape),
        "train_label_distribution": {str(k): int(v) for k, v in Counter(train_y_int.tolist()).items()},
        "test_label_distribution": {str(k): int(v) for k, v in Counter(test_y_int.tolist()).items()},
        "train_label_accuracy_vs_official": float(train_label_accuracy),
        "train_label_macro_f1_vs_official": float(train_label_macro_f1),
        "test_label_accuracy_vs_official": float(test_label_accuracy),
        "test_label_macro_f1_vs_official": float(test_label_macro_f1),
        "best_epoch": int(tracker.best["epoch"]),
        "best_test_accuracy": float(tracker.best["accuracy"]),
        "best_test_macro_f1": float(tracker.best["macro_f1"]),
        "best_test_weighted_f1": float(tracker.best["weighted_f1"]),
        "final_test_accuracy": float(accuracy_score(test_y_int, pred_int)),
        "final_test_macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "final_test_weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
        "classification_report": report_dict,
        "confusion_matrix": confusion_matrix(test_y_int, pred_int, labels=[0, 1, 2]).tolist(),
        "epoch_rows": tracker.rows,
        "checkpoint_path": str(checkpoint_path),
    }

    result_path = result_dir / f"{args.run_name}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "run_name": args.run_name,
            "formula": args.formula,
            "horizon_steps": args.horizon_steps,
            "alpha": args.alpha,
            "best_epoch": result["best_epoch"],
            "best_test_accuracy": result["best_test_accuracy"],
            "best_test_macro_f1": result["best_test_macro_f1"],
            "best_test_weighted_f1": result["best_test_weighted_f1"],
            "test_label_accuracy_vs_official": result["test_label_accuracy_vs_official"],
            "test_label_macro_f1_vs_official": result["test_label_macro_f1_vs_official"],
            "result_path": str(result_path),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
