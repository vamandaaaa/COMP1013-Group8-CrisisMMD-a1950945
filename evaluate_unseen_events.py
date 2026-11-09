"""Evaluate CrisisMMD on disasters excluded from training and validation.

Uses cleaned train/dev only. Never reads the original test split.
Run alongside the existing train_models.py.
"""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler

from train_models import (
    read_split, read_features, features_for_rows,
    train_model, predict_probabilities as probabilities
)


def assess(y, pred, labels):
    all_classes = list(range(len(labels)))
    present_classes = sorted(set(y.tolist()))

    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1_all": round(float(f1_score(
            y, pred, labels=all_classes,
            average="macro", zero_division=0
        )), 4),
        "macro_f1_present": round(float(f1_score(
            y, pred, labels=present_classes,
            average="macro", zero_division=0
        )), 4),
        "per_class": classification_report(
            y, pred, labels=all_classes,
            target_names=labels, output_dict=True,
            zero_division=0
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="CrisisMMD leave-one-disaster-out experiment"
    )
    parser.add_argument("--processed-dir", type=Path,
                        default=Path("data/processed"))
    parser.add_argument("--group-zip", type=Path,
                        default=Path("crisismmd_group_share.zip"))
    parser.add_argument("--results-dir", type=Path,
                        default=Path("results/unseen_events"))
    parser.add_argument("--figure", type=Path,
                        default=Path("figures/unseen_events/macro_f1.png"))
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable. Use --device cpu")

    device = torch.device(args.device)

    # The original test split is not read.
    original_train = read_split(
        args.processed_dir / "humanitarian_train.tsv"
    )
    original_dev = read_split(
        args.processed_dir / "humanitarian_dev.tsv"
    )

    if not original_train or not original_dev:
        parser.error("Cleaned train/dev files are missing or empty")

    rows = original_train + original_dev
    labels = sorted({r["label"] for r in rows})
    events = sorted({r["event_name"] for r in rows})

    if len(labels) != 5 or len(events) < 2:
        parser.error("Expected five classes and multiple disaster events")

    for key in ("tweet_id", "image_id"):
        train_ids = {r[key] for r in original_train}
        dev_ids = {r[key] for r in original_dev}
        if train_ids & dev_ids:
            raise ValueError(f"Original train/dev share {key}")

    with ZipFile(args.group_zip) as archive:
        text_ids, text_features = read_features(
            archive, "text_bert.npz", "tweet_id"
        )
        image_ids, image_features = read_features(
            archive, "image_b0.npz", "image_id"
        )

    image_hashes = {}

    for r in rows:
        image_id = r["image_id"]

        if image_id not in image_ids:
            raise ValueError("Missing image feature: " + image_id)

        if image_id not in image_hashes:
            feature = image_features[image_ids[image_id]]
            image_hashes[image_id] = hashlib.sha256(
                feature.tobytes()
            ).hexdigest()

    folds = []
    print("Original official test: NOT USED")

    for event in events:
        heldout = [
            r for r in rows if r["event_name"] == event
        ]

        held_tweets = {r["tweet_id"] for r in heldout}
        held_images = {r["image_id"] for r in heldout}
        held_hashes = {
            image_hashes[r["image_id"]] for r in heldout
        }

        def not_leaking(r):
            return (
                r["event_name"] != event
                and r["tweet_id"] not in held_tweets
                and r["image_id"] not in held_images
                and image_hashes[r["image_id"]] not in held_hashes
            )

        train = [
            r for r in original_train if not_leaking(r)
        ]
        dev = [
            r for r in original_dev if not_leaking(r)
        ]

        if not train or not dev:
            raise ValueError("Empty train/dev fold for " + event)

        if {r["label"] for r in train} != set(labels):
            raise ValueError("Training fold missing a class: " + event)

        n_classes = len({r["label"] for r in heldout})

        print(
            f"{event}: train {len(train)}, dev {len(dev)}, "
            f"unseen {len(heldout)}, unseen classes {n_classes}"
        )

        folds.append((event, train, dev, heldout))

    if args.check_only:
        print("Split check passed. No models trained.")
        return

    label_ids = {
        label: i for i, label in enumerate(labels)
    }

    model_names = [
        "text_only", "image_only",
        "early_fusion", "late_fusion"
    ]

    all_true = []
    all_predictions = {
        name: [] for name in model_names
    }
    results = []

    for event, train, dev, heldout in folds:
        print("\nUnseen event:", event)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        yt = np.array(
            [label_ids[r["label"]] for r in train],
            dtype=np.int64
        )
        yd = np.array(
            [label_ids[r["label"]] for r in dev],
            dtype=np.int64
        )
        yh = np.array(
            [label_ids[r["label"]] for r in heldout],
            dtype=np.int64
        )

        def scaled_features(key, lookup, cache):
            a = features_for_rows(train, key, lookup, cache)
            b = features_for_rows(dev, key, lookup, cache)
            c = features_for_rows(heldout, key, lookup, cache)

            scaler = StandardScaler().fit(a)

            return tuple(
                scaler.transform(x).astype(np.float32)
                for x in (a, b, c)
            )

        xt = scaled_features(
            "tweet_id", text_ids, text_features
        )
        xi = scaled_features(
            "image_id", image_ids, image_features
        )

        inputs = {
            "text_only": xt,
            "image_only": xi,
            "early_fusion": tuple(
                np.concatenate((a, b), axis=1)
                for a, b in zip(xt, xi)
            ),
        }

        counts = np.bincount(
            yt, minlength=len(labels)
        )
        weights = np.sqrt(
            len(yt) / (len(labels) * counts)
        )
        weights /= weights.mean()

        dev_probs = {}
        unseen_probs = {}
        epochs = {}

        for name, (a, b, c) in inputs.items():
            model, scores, history, dp = train_model(
                name, a, yt, b, yd,
                len(labels), weights, args, device
            )

            dev_probs[name] = dp
            unseen_probs[name] = probabilities(
                model, c, device
            )
            epochs[name] = scores["best_epoch"]

        best_alpha = 0.0
        best_f1 = -1.0

        for alpha in np.linspace(0, 1, 21):
            p = (
                alpha * dev_probs["text_only"]
                + (1 - alpha) * dev_probs["image_only"]
            )

            f1 = f1_score(
                yd, p.argmax(1),
                labels=list(range(len(labels))),
                average="macro",
                zero_division=0
            )

            if f1 > best_f1:
                best_alpha = float(alpha)
                best_f1 = float(f1)

        unseen_probs["late_fusion"] = (
            best_alpha * unseen_probs["text_only"]
            + (1 - best_alpha) * unseen_probs["image_only"]
        )

        fold = {
            "event": event,
            "samples": len(heldout),
            "class_support": dict(
                Counter(r["label"] for r in heldout)
            ),
            "best_epochs": epochs,
            "late_text_weight": round(best_alpha, 2),
            "models": {},
        }

        all_true.extend(yh.tolist())

        for name, p in unseen_probs.items():
            pred = p.argmax(1)
            all_predictions[name].extend(
                pred.tolist()
            )

            fold["models"][name] = assess(
                yh, pred, labels
            )

            print(
                name, "event macro-F1",
                fold["models"][name]["macro_f1_present"]
            )

        results.append(fold)

    args.results_dir.mkdir(
        parents=True, exist_ok=True
    )

    summary = {
        "design": (
            "leave-one-event-out; uses cleaned train/dev only; "
            "original test not used"
        ),
        "settings": {
            "dropout": args.dropout,
            "lr": args.lr,
            "seed": args.seed,
            "epochs": args.epochs,
            "patience": args.patience,
        },
        "limitation": (
            "Feature-exact duplicates are screened, "
            "but some near-duplicate images may remain."
        ),
        "labels": labels,
        "events": results,
        "pooled": {
            name: assess(
                np.array(all_true),
                np.array(all_predictions[name]),
                labels
            )
            for name in model_names
        },
    }

    output = args.results_dir / "summary.json"
    output.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8"
    )

    csv_path = args.results_dir / "event_scores.csv"

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(
            f, lineterminator="\n"
        )

        writer.writerow(
            ["event", "samples"] + model_names
        )

        for result in results:
            writer.writerow(
                [result["event"], result["samples"]]
                + [
                    result["models"][name]["macro_f1_present"]
                    for name in model_names
                ]
            )

    x = np.arange(len(results))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, name in enumerate(model_names):
        values = [
            result["models"][name]["macro_f1_present"]
            for result in results
        ]

        ax.bar(
            x + (i - 1.5) * width,
            values,
            width,
            label=name.replace("_", " ")
        )

    ax.set_xticks(
        x,
        [result["event"].replace("_", " ") for result in results],
        rotation=35,
        ha="right"
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel(
        "Macro-F1 (classes present in each event)"
    )
    ax.set_title(
        "Evaluation on held-out disaster events"
    )
    ax.legend()

    fig.tight_layout()

    args.figure.parent.mkdir(
        parents=True, exist_ok=True
    )
    fig.savefig(args.figure, dpi=170)
    plt.close(fig)

    print("\nPooled leave-one-event-out macro-F1:")

    for name, value in summary["pooled"].items():
        print(name, value["macro_f1_all"])

    print("Saved:", output)
    print("Original official test: NOT USED")


if __name__ == "__main__":
    main()
