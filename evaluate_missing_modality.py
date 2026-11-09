"""Test fusion models with zeroed text or image features.

Uses development data only, not the official test set.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from train_cross_attention import (
    CrossAttentionClassifier, load_cache, read_split
)
from train_pooled_fusion import PooledFusionClassifier


def predict(model, text, masks, patches, device,
            batch_size, condition):
    model.eval()
    predictions = []

    with torch.inference_mode():
        for start in range(0, len(text), batch_size):
            end = start + batch_size

            t = torch.from_numpy(
                text[start:end]
            ).to(device).float()

            m = torch.from_numpy(
                masks[start:end]
            ).to(device)

            p = torch.from_numpy(
                patches[start:end]
            ).to(device).float()

            if condition == "zero_text":
                t = torch.zeros_like(t)
            elif condition == "zero_image":
                p = torch.zeros_like(p)

            output = model(t, m, p)
            predictions.extend(
                output.argmax(dim=1).cpu().tolist()
            )

    return np.array(predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-dir", type=Path,
        default=Path("data/processed")
    )
    parser.add_argument(
        "--features-dir", type=Path,
        default=Path("data/features")
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "results/robustness/missing_modality_dev.csv"
        )
    )
    parser.add_argument(
        "--device", choices=["cpu", "mps"],
        default="cpu"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.device == "mps":
        if not torch.backends.mps.is_available():
            parser.error("MPS unavailable")

    if args.batch_size < 1:
        parser.error("Batch size must be positive")

    dev = read_split(
        args.processed_dir / "humanitarian_dev.tsv"
    )

    if not dev:
        raise ValueError("Empty development split")

    labels = sorted({r["label"] for r in dev})
    if len(labels) != 5:
        raise ValueError("Expected five classes")

    label_to_int = {
        label: i for i, label in enumerate(labels)
    }

    y = np.array([
        label_to_int[r["label"]] for r in dev
    ])

    affected_id = label_to_int["affected_individuals"]

    text_ids, text_features, text_masks = load_cache(
        args.features_dir / "bert_token_features.npz",
        "tweet_id", "feat"
    )

    image_ids, image_features, _ = load_cache(
        args.features_dir / "vit_b_16_features.npz",
        "image_id", "patch_feat"
    )

    text_indices = [
        text_ids[r["tweet_id"]] for r in dev
    ]

    image_indices = [
        image_ids[r["image_id"]] for r in dev
    ]

    text = text_features[text_indices]
    masks = text_masks[text_indices]
    patches = image_features[image_indices]

    trials = []

    for seed in (42, 43, 44):
        if seed == 42:
            pooled = (
                "models/pooled_fusion/grid_seed42/"
                "dropout_0.3_lr_0.001/pooled_fusion.pt"
            )
            attention = (
                "models/cross_attention/grid_seed42/"
                "dropout_0.3_lr_0.003/cross_attention.pt"
            )
        else:
            pooled = (
                f"models/pooled_fusion/"
                f"best_config_seed{seed}/pooled_fusion.pt"
            )
            attention = (
                f"models/cross_attention/"
                f"best_config_seed{seed}/cross_attention.pt"
            )

        trials.append(
            ("pooled_fusion", seed, Path(pooled))
        )
        trials.append(
            ("cross_attention", seed, Path(attention))
        )

    for name, seed, path in trials:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint: {path}"
            )

    device = torch.device(args.device)
    rows = []
    conditions = ("both", "zero_image", "zero_text")

    for name, seed, path in trials:
        if name == "pooled_fusion":
            model = PooledFusionClassifier(
                hidden=128, dropout=0.3
            )
        else:
            model = CrossAttentionClassifier(
                hidden=128, heads=4, dropout=0.3
            )

        model.load_state_dict(
            torch.load(
                path, map_location="cpu",
                weights_only=True
            )
        )
        model.to(device).eval()

        for condition in conditions:
            pred = predict(
                model, text, masks, patches,
                device, args.batch_size, condition
            )

            score = f1_score(
                y, pred, labels=range(5),
                average="macro",
                zero_division=0
            )

            affected_recall = float(
                np.mean(
                    pred[y == affected_id] == affected_id
                )
            )

            rows.append({
                "model": name,
                "seed": seed,
                "condition": condition,
                "macro_f1": round(float(score), 4),
                "affected_recall": round(
                    affected_recall, 4
                )
            })

            print(
                f"{name:16} seed={seed} "
                f"{condition:12} F1={score:.4f}"
            )

    args.output.parent.mkdir(
        parents=True, exist_ok=True
    )

    with args.output.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nMean macro-F1 over 3 seeds:")

    for name in ("pooled_fusion", "cross_attention"):
        for condition in conditions:
            values = [
                r["macro_f1"] for r in rows
                if r["model"] == name
                and r["condition"] == condition
            ]

            print(
                f"{name:16} {condition:12} "
                f"{np.mean(values):.4f}"
            )

    print("\nSaved:", args.output)
    print("Official test: NOT ACCESSED")
    print("Zero features are an artificial stress test.")


if __name__ == "__main__":
    main()
