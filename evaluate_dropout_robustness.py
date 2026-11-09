"""Compare pooled fusion with and without modality dropout on dev."""

import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from evaluate_missing_modality import predict
from train_cross_attention import load_cache, read_split
from train_pooled_fusion import PooledFusionClassifier


def checkpoint(model_name, seed):
    if model_name == "modality_dropout":
        return Path(
            f"models/modality_dropout/seed{seed}/modality_dropout.pt"
        )

    if seed == 42:
        return Path(
            "models/pooled_fusion/grid_seed42/"
            "dropout_0.3_lr_0.001/pooled_fusion.pt"
        )

    return Path(
        f"models/pooled_fusion/"
        f"best_config_seed{seed}/pooled_fusion.pt"
    )


def main():
    device = (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    dev = read_split(
        Path("data/processed/humanitarian_dev.tsv")
    )

    labels = sorted({row["label"] for row in dev})

    if len(labels) != 5:
        raise ValueError("Expected five classes")

    label_to_int = {
        label: i for i, label in enumerate(labels)
    }

    y = np.array([
        label_to_int[row["label"]] for row in dev
    ])

    text_lookup, text_features, masks = load_cache(
        Path("data/features/bert_token_features.npz"),
        "tweet_id", "feat"
    )

    image_lookup, image_features, _ = load_cache(
        Path("data/features/vit_b_16_features.npz"),
        "image_id", "patch_feat"
    )

    text_indices = [
        text_lookup[row["tweet_id"]] for row in dev
    ]

    image_indices = [
        image_lookup[row["image_id"]] for row in dev
    ]

    text = text_features[text_indices]
    masks = masks[text_indices]
    images = image_features[image_indices]

    results = []

    for name in ("pooled_fusion", "modality_dropout"):
        for seed in (42, 43, 44):
            path = checkpoint(name, seed)

            if not path.is_file():
                raise FileNotFoundError(path)

            model = PooledFusionClassifier(
                hidden=128, dropout=0.3
            )

            model.load_state_dict(
                torch.load(
                    path,
                    map_location="cpu",
                    weights_only=True
                )
            )

            model = model.to(device).eval()

            for condition in ("both", "zero_image", "zero_text"):
                pred = predict(
                    model, text, masks, images,
                    device, 32, condition
                )

                score = f1_score(
                    y, pred,
                    labels=range(5),
                    average="macro",
                    zero_division=0
                )

                results.append({
                    "model": name,
                    "seed": seed,
                    "condition": condition,
                    "macro_f1": round(float(score), 6)
                })

                print(
                    f"{name:16} seed={seed} "
                    f"{condition:10} F1={score:.4f}"
                )

    output = Path(
        "results/robustness/dropout_comparison_dev.csv"
    )

    output.parent.mkdir(
        parents=True, exist_ok=True
    )

    with output.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(results[0])
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nMean development macro-F1 across three seeds:")

    for condition in ("both", "zero_image", "zero_text"):
        averages = {}

        for name in ("pooled_fusion", "modality_dropout"):
            scores = [
                row["macro_f1"] for row in results
                if row["model"] == name
                and row["condition"] == condition
            ]

            averages[name] = float(np.mean(scores))

        difference = (
            averages["modality_dropout"]
            - averages["pooled_fusion"]
        )

        print(
            f"{condition:10} "
            f"baseline={averages['pooled_fusion']:.4f} "
            f"dropout={averages['modality_dropout']:.4f} "
            f"difference={difference:+.4f}"
        )

    print("\nSaved:", output)
    print("Official test: NOT ACCESSED")


if __name__ == "__main__":
    main()
