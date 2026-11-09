"""Compare frozen image encoders on cleaned CrisisMMD train/dev splits.

Uses the same BERT features and classifier for each image encoder.
The official test split is not accessed.
"""

import argparse
import csv
import json
import random
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from train_models import (
    read_split,
    read_features,
    features_for_rows,
    train_model
)


DIMENSIONS = {
    "efficientnet_b0": 1280,
    "resnet50": 2048,
    "vit_b_16": 768
}


def load_new_features(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run full image extraction first."
        )

    with np.load(path, allow_pickle=False) as data:
        ids = [str(x) for x in data["image_id"]]
        features = data["feat"].astype(np.float32)

    if len(ids) != len(set(ids)) or len(ids) != len(features):
        raise ValueError(f"Duplicate or mismatched IDs: {path}")

    lookup = {
        image_id: i for i, image_id in enumerate(ids)
    }

    return lookup, features


def check_features(rows, key, lookup, features, name, dimension):
    if features.ndim != 2 or features.shape[1] != dimension:
        raise ValueError(
            f"{name}: expected {dimension} dimensions, "
            f"got {features.shape}"
        )

    result = features_for_rows(rows, key, lookup, features)

    if not np.isfinite(result).all():
        raise ValueError(f"{name}: invalid feature values")

    print(
        f"{name}: {len(result)} aligned rows, "
        f"{dimension} dimensions"
    )

    return result


def scale_splits(train, dev):
    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(train).astype(np.float32)
    dev_scaled = scaler.transform(dev).astype(np.float32)

    return train_scaled, dev_scaled, scaler


def fit_one(
    name, x_train, x_dev,
    y_train, y_dev, weights, args, device
):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, metrics, history, dev_probs = train_model(
        name,
        x_train,
        y_train,
        x_dev,
        y_dev,
        5,
        weights,
        args,
        device
    )

    torch.save(
        model.state_dict(),
        args.models_dir / f"{name}.pt"
    )

    return metrics, history, dev_probs


def main():
    parser = argparse.ArgumentParser(
        description="Compare CrisisMMD image encoders"
    )

    parser.add_argument(
        "--processed-dir", type=Path,
        default=Path("data/processed")
    )
    parser.add_argument(
        "--group-zip", type=Path,
        default=Path("crisismmd_group_share.zip")
    )
    parser.add_argument(
        "--features-dir", type=Path,
        default=Path("data/features")
    )
    parser.add_argument(
        "--results-dir", type=Path, default=None
    )
    parser.add_argument(
        "--models-dir", type=Path, default=None
    )
    parser.add_argument(
        "--device", choices=["cpu", "mps"], default="cpu"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.results_dir is None:
        args.results_dir = Path(
            f"results/encoder_comparison/seed{args.seed}"
        )

    if args.models_dir is None:
        args.models_dir = Path(
            f"models/encoder_comparison/seed{args.seed}"
        )

    if not 0 <= args.dropout < 1 or args.lr <= 0:
        parser.error("Invalid dropout or learning rate")

    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS unavailable. Use --device cpu")

    # Read cleaned training and development splits only.
    train = read_split(
        args.processed_dir / "humanitarian_train.tsv"
    )
    dev = read_split(
        args.processed_dir / "humanitarian_dev.tsv"
    )

    if not train or not dev:
        parser.error("Cleaned train/dev files are missing or empty")

    labels = sorted({r["label"] for r in train})

    if len(labels) != 5 or {r["label"] for r in dev} != set(labels):
        raise ValueError("Expected five humanitarian classes")

    rows = train + dev

    for key in ("tweet_id", "image_id"):
        train_ids = {r[key] for r in train}
        dev_ids = {r[key] for r in dev}

        if train_ids & dev_ids:
            raise ValueError(f"Train/dev overlap in {key}")

    print("Train:", len(train), "Dev:", len(dev))
    print("Original official test: NOT ACCESSED")

    # Existing frozen BERT and EfficientNet-B0 features.
    with ZipFile(args.group_zip) as archive:
        text_lookup, text_features = read_features(
            archive, "text_bert.npz", "tweet_id"
        )

        b0_lookup, b0_features = read_features(
            archive, "image_b0.npz", "image_id"
        )

    raw_text = check_features(
        rows, "tweet_id", text_lookup,
        text_features, "BERT", 768
    )

    raw_images = {
        "efficientnet_b0": check_features(
            rows, "image_id", b0_lookup,
            b0_features, "EfficientNet-B0", 1280
        )
    }

    # Newly extracted ResNet-50 and ViT features.
    for name in ("resnet50", "vit_b_16"):
        path = args.features_dir / f"{name}_features.npz"
        lookup, features = load_new_features(path)

        raw_images[name] = check_features(
            rows, "image_id", lookup,
            features, name, DIMENSIONS[name]
        )

    if args.check_only:
        print("All encoder feature checks passed.")
        print("No training performed.")
        return

    output_file = args.results_dir / "dev_results.json"

    if output_file.exists() and not args.overwrite:
        parser.error(
            "Results already exist. Use --overwrite to rerun."
        )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    label_ids = {
        label: i for i, label in enumerate(labels)
    }

    y_train = np.array(
        [label_ids[r["label"]] for r in train],
        dtype=np.int64
    )
    y_dev = np.array(
        [label_ids[r["label"]] for r in dev],
        dtype=np.int64
    )

    counts = np.bincount(y_train, minlength=5)

    weights = np.sqrt(
        len(y_train) / (5 * counts)
    )
    weights /= weights.mean()

    n_train = len(train)

    # Fit scalers on training data only.
    xt_train, xt_dev, text_scaler = scale_splits(
        raw_text[:n_train],
        raw_text[n_train:]
    )

    scaler_data = {
        "text_mean": text_scaler.mean_,
        "text_scale": text_scaler.scale_
    }

    all_history = []
    records = []
    results = {}

    # Train the shared text baseline once.
    print("\nTraining text-only baseline...")

    text_metrics, history, text_probs = fit_one(
        "text_only",
        xt_train, xt_dev,
        y_train, y_dev,
        weights, args, device
    )

    all_history.extend(history)
    results["text_only"] = text_metrics

    records.append({
        "encoder": "BERT",
        "approach": "text_only",
        "dev_macro_f1": text_metrics["dev"]["macro_f1"],
        "dev_accuracy": text_metrics["dev"]["accuracy"],
        "best_epoch": text_metrics["best_epoch"],
        "text_weight": ""
    })

    # Compare the three image encoders.
    for encoder, image_rows in raw_images.items():
        print(f"\nTraining with image encoder: {encoder}")

        xi_train, xi_dev, image_scaler = scale_splits(
            image_rows[:n_train],
            image_rows[n_train:]
        )

        scaler_data[f"{encoder}_mean"] = image_scaler.mean_
        scaler_data[f"{encoder}_scale"] = image_scaler.scale_

        early_train = np.concatenate(
            [xt_train, xi_train], axis=1
        )
        early_dev = np.concatenate(
            [xt_dev, xi_dev], axis=1
        )

        image_name = f"{encoder}_image_only"
        early_name = f"{encoder}_early_fusion"

        # Image-only classifier.
        image_metrics, image_history, image_probs = fit_one(
            image_name,
            xi_train, xi_dev,
            y_train, y_dev,
            weights, args, device
        )

        # Early fusion classifier.
        early_metrics, early_history, early_probs = fit_one(
            early_name,
            early_train, early_dev,
            y_train, y_dev,
            weights, args, device
        )

        all_history.extend(image_history + early_history)

        results[image_name] = image_metrics
        results[early_name] = early_metrics

        for approach, metrics in (
            ("image_only", image_metrics),
            ("early_fusion", early_metrics)
        ):
            records.append({
                "encoder": encoder,
                "approach": approach,
                "dev_macro_f1": metrics["dev"]["macro_f1"],
                "dev_accuracy": metrics["dev"]["accuracy"],
                "best_epoch": metrics["best_epoch"],
                "text_weight": ""
            })

        # Late fusion: select alpha using dev only.
        best_alpha = 0.0
        best_f1 = -1.0

        for alpha in np.linspace(0, 1, 21):
            probs = (
                alpha * text_probs
                + (1 - alpha) * image_probs
            )

            score = f1_score(
                y_dev,
                probs.argmax(axis=1),
                labels=list(range(5)),
                average="macro",
                zero_division=0
            )

            if score > best_f1:
                best_f1 = float(score)
                best_alpha = float(alpha)

        late_probs = (
            best_alpha * text_probs
            + (1 - best_alpha) * image_probs
        )

        late_pred = late_probs.argmax(axis=1)
        late_accuracy = float(np.mean(late_pred == y_dev))

        results[f"{encoder}_late_fusion"] = {
            "dev": {
                "macro_f1": round(best_f1, 4),
                "accuracy": round(late_accuracy, 4)
            },
            "text_probability_weight": round(best_alpha, 2),
            "note": "Alpha selected on dev; no separate network"
        }

        records.append({
            "encoder": encoder,
            "approach": "late_fusion",
            "dev_macro_f1": round(best_f1, 4),
            "dev_accuracy": round(late_accuracy, 4),
            "best_epoch": "",
            "text_weight": round(best_alpha, 2)
        })

    # Save comparison table.
    comparison_file = args.results_dir / "comparison.csv"

    with comparison_file.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "encoder", "approach", "dev_macro_f1",
                "dev_accuracy", "best_epoch", "text_weight"
            ]
        )
        writer.writeheader()
        writer.writerows(records)

    # Save training curves as data.
    history_file = args.results_dir / "training_history.csv"

    with history_file.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(all_history[0])
        )
        writer.writeheader()
        writer.writerows(all_history)

    np.savez_compressed(
        args.models_dir / "feature_scalers.npz",
        **scaler_data
    )

    config = {
        "description": "Frozen image encoder comparison",
        "official_test_accessed": False,
        "seed": args.seed,
        "device": args.device,
        "epochs_max": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "learning_rate": args.lr,
        "labels": labels,
        "split_sizes": {
            "train": n_train,
            "dev": len(dev)
        },
        "encoder_dimensions": DIMENSIONS,
        "class_weights": weights.tolist(),
        "results": results
    }

    output_file.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8"
    )

    print("\nDevelopment comparison:")

    for row in records:
        print(
            f"{row['encoder']:16} "
            f"{row['approach']:14} "
            f"macro-F1 {row['dev_macro_f1']:.4f}"
        )

    print("\nSaved:", comparison_file)
    print("Saved:", output_file)
    print("Official test: NOT ACCESSED")


if __name__ == "__main__":
    main()
