"""Train cross-attention using frozen BERT and ViT features.

Uses cleaned train and dev only. Official test is not used.
"""

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class PooledFusionClassifier(nn.Module):
    """Combine pooled BERT tokens and ViT image patches."""

    def __init__(self, hidden=128, heads=4, dropout=0.5):
        super().__init__()

        self.text_proj = nn.Linear(768, hidden)
        self.image_proj = nn.Linear(768, hidden)
        self.norm = nn.LayerNorm(hidden)

        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 5)
        )

    def forward(self, text, mask, patches):
        text = self.text_proj(text)
        image = self.image_proj(patches)

        # Average valid text tokens only.
        valid = mask.unsqueeze(-1).float()
        count = valid.sum(dim=1).clamp(min=1)
        text_mean = (text * valid).sum(dim=1) / count

        # Average image patches, without attention.
        image_mean = image.mean(dim=1)

        fused = self.norm(
            text + image_mean.unsqueeze(1)
        )
        fused_mean = (fused * valid).sum(dim=1) / count

        combined = torch.cat(
            [text_mean, fused_mean, image_mean], dim=1
        )

        return self.classifier(combined)



def read_split(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_cache(path, id_key, feature_key):
    if not path.is_file():
        raise FileNotFoundError(f"Feature cache not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        ids = [str(x) for x in data[id_key]]
        features = data[feature_key].copy()

        masks = (
            data["attention_mask"].copy()
            if "attention_mask" in data else None
        )

    if len(ids) != len(set(ids)) or len(features) != len(ids):
        raise ValueError(f"Invalid or repeated IDs in {path}")

    lookup = {key: i for i, key in enumerate(ids)}
    return lookup, features, masks


def make_loader(rows, labels, text_lookup, text_features,
                masks, image_lookup, image_features,
                batch_size, shuffle, seed):

    text_indices = [
        text_lookup[r["tweet_id"]] for r in rows
    ]
    image_indices = [
        image_lookup[r["image_id"]] for r in rows
    ]

    y = np.array(
        [labels[r["label"]] for r in rows],
        dtype=np.int64
    )

    dataset = TensorDataset(
        torch.from_numpy(text_features[text_indices]),
        torch.from_numpy(masks[text_indices]),
        torch.from_numpy(image_features[image_indices]),
        torch.from_numpy(y)
    )

    generator = torch.Generator().manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0
    )

    return loader, y


def evaluate(model, loader, criterion, device):
    model.eval()

    all_true = []
    all_pred = []
    total_loss = 0.0

    with torch.no_grad():
        for text, mask, image, y in loader:
            logits = model(
                text.to(device).float(),
                mask.to(device),
                image.to(device).float()
            )

            loss = criterion(logits, y.to(device))
            total_loss += loss.item() * len(y)

            all_true.extend(y.tolist())
            all_pred.extend(logits.argmax(1).cpu().tolist())

    scores = {
        "loss": total_loss / len(loader.dataset),
        "accuracy": float(accuracy_score(all_true, all_pred)),
        "macro_f1": float(f1_score(
            all_true, all_pred,
            labels=list(range(5)),
            average="macro",
            zero_division=0
        )),
        "weighted_f1": float(f1_score(
            all_true, all_pred,
            labels=list(range(5)),
            average="weighted",
            zero_division=0
        ))
    }

    return scores, np.array(all_true), np.array(all_pred)


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
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--modality-dropout", type=float, default=0.15)
    args = parser.parse_args()

    if not 0 <= args.modality_dropout <= 0.5:
        parser.error("Invalid modality dropout rate")

    if args.hidden % args.heads != 0:
        parser.error("Hidden size must be divisible by attention heads")

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("Invalid training settings")

    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("Apple MPS is unavailable")

    if args.results_dir is None:
        args.results_dir = Path(
            f"results/modality_dropout/seed{args.seed}"
        )

    if args.models_dir is None:
        args.models_dir = Path(
            f"models/modality_dropout/seed{args.seed}"
        )

    train = read_split(
        args.processed_dir / "humanitarian_train.tsv"
    )
    dev = read_split(
        args.processed_dir / "humanitarian_dev.tsv"
    )

    if not train or not dev:
        raise ValueError("Train and dev must not be empty")

    labels = sorted(set(r["label"] for r in train))

    if len(labels) != 5 or set(r["label"] for r in dev) != set(labels):
        raise ValueError("Expected five classes")

    for key in ("tweet_id", "image_id"):
        if {r[key] for r in train} & {r[key] for r in dev}:
            raise ValueError(f"Train/dev overlap: {key}")

    text_lookup, text_features, masks = load_cache(
        args.features_dir / "bert_token_features.npz",
        "tweet_id", "feat"
    )

    image_lookup, image_features, _ = load_cache(
        args.features_dir / "vit_b_16_features.npz",
        "image_id", "patch_feat"
    )

    if (
        text_features.ndim != 3
        or text_features.shape[1:] != (48, 768)
        or image_features.ndim != 3
        or image_features.shape[1:] != (16, 768)
        or masks is None
        or masks.shape != text_features.shape[:2]
    ):
        raise ValueError("Unexpected BERT or ViT feature dimensions")

    if any(
        r["tweet_id"] not in text_lookup
        or r["image_id"] not in image_lookup
        for r in train + dev
    ):
        raise ValueError("Missing features for train/dev rows")

    print(f"Train: {len(train)} | Dev: {len(dev)}")
    print("BERT tokens:", text_features.shape)
    print("ViT patches:", image_features.shape)
    print("Original official test: NOT ACCESSED")

    # Test one forward pass before training.
    model = PooledFusionClassifier(
        args.hidden, args.heads, args.dropout
    )

    t = text_lookup[train[0]["tweet_id"]]
    i = image_lookup[train[0]["image_id"]]

    with torch.no_grad():
        output = model(
            torch.from_numpy(text_features[[t]]).float(),
            torch.from_numpy(masks[[t]]),
            torch.from_numpy(image_features[[i]]).float()
        )

    if output.shape != (1, 5) or not torch.isfinite(output).all():
        raise ValueError("Pooled fusion forward check failed")

    print("Pooled fusion forward check passed")

    if args.check_only:
        print("No training performed")
        return

    results_file = args.results_dir / "dev_results.json"

    if results_file.exists() and not args.overwrite:
        parser.error("Results already exist. Use --overwrite to rerun")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    model = PooledFusionClassifier(
        args.hidden, args.heads, args.dropout
    ).to(device)

    label_ids = {name: i for i, name in enumerate(labels)}

    train_loader, y_train = make_loader(
        train, label_ids, text_lookup, text_features,
        masks, image_lookup, image_features,
        args.batch_size, True, args.seed
    )

    train_eval_loader, _ = make_loader(
        train, label_ids, text_lookup, text_features,
        masks, image_lookup, image_features,
        args.batch_size, False, args.seed
    )

    dev_loader, _ = make_loader(
        dev, label_ids, text_lookup, text_features,
        masks, image_lookup, image_features,
        args.batch_size, False, args.seed
    )

    counts = np.bincount(y_train, minlength=5)

    weights = np.sqrt(len(y_train) / (5 * counts))
    weights /= weights.mean()

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            weights, dtype=torch.float32, device=device
        )
    )

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    history = []
    best_state = None
    best_epoch = 0
    best_f1 = -1.0
    best_loss = float("inf")
    stagnant = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        for text, mask, image, y in train_loader:
            text = text.to(device).float()
            mask = mask.to(device)
            image = image.to(device).float()
            y = y.to(device)

            # Randomly remove one modality during training.
            r = torch.rand(len(y), device=device)
            drop_text = r < args.modality_dropout
            drop_image = (
                (r >= args.modality_dropout)
                & (r < 2 * args.modality_dropout)
            )

            text = text.masked_fill(
                drop_text[:, None, None], 0.0
            )
            image = image.masked_fill(
                drop_image[:, None, None], 0.0
            )

            optimiser.zero_grad()
            logits = model(text, mask, image)
            loss = criterion(logits, y)
            loss.backward()
            optimiser.step()

        train_scores, _, _ = evaluate(
            model, train_eval_loader, criterion, device
        )

        dev_scores, _, _ = evaluate(
            model, dev_loader, criterion, device
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_scores["loss"], 5),
            "dev_loss": round(dev_scores["loss"], 5),
            "train_macro_f1": round(train_scores["macro_f1"], 5),
            "dev_macro_f1": round(dev_scores["macro_f1"], 5)
        })

        print(
            f"Epoch {epoch:02d} | "
            f"train F1 {train_scores['macro_f1']:.4f} | "
            f"dev F1 {dev_scores['macro_f1']:.4f}"
        )

        f1 = dev_scores["macro_f1"]
        dev_loss = dev_scores["loss"]

        if (
            f1 > best_f1 + 1e-6
            or (
                abs(f1 - best_f1) <= 1e-6
                and dev_loss < best_loss
            )
        ):
            best_f1 = f1
            best_loss = dev_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stagnant = 0
        else:
            stagnant += 1

            if stagnant >= args.patience:
                break

    model.load_state_dict(best_state)

    train_scores, _, _ = evaluate(
        model, train_eval_loader, criterion, device
    )

    dev_scores, y_true, predicted = evaluate(
        model, dev_loader, criterion, device
    )

    dev_scores["per_class"] = classification_report(
        y_true,
        predicted,
        target_names=labels,
        output_dict=True,
        zero_division=0
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(),
        args.models_dir / "modality_dropout.pt"
    )

    summary = {
        "model": "BERT token and ViT patch pooled fusion with modality dropout",
        "seed": args.seed,
        "training_samples": len(train),
        "development_samples": len(dev),
        "selected_epoch": best_epoch,
        "hidden": args.hidden,
        "heads": None,
        "dropout": args.dropout,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "labels": labels,
        "modality_dropout": args.modality_dropout,
        "class_weights": weights.tolist(),
        "train": train_scores,
        "dev": dev_scores,
        "official_test_accessed": False
    }

    results_file.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8"
    )

    history_file = args.results_dir / "training_history.csv"

    with history_file.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(history[0])
        )
        writer.writeheader()
        writer.writerows(history)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        [h["epoch"] for h in history],
        [h["train_macro_f1"] for h in history],
        label="Train"
    )

    ax.plot(
        [h["epoch"] for h in history],
        [h["dev_macro_f1"] for h in history],
        label="Development"
    )

    ax.set(
        xlabel="Epoch",
        ylabel="Macro-F1",
        title="Pooled fusion learning curve"
    )

    ax.legend()
    fig.tight_layout()

    fig.savefig(
        args.results_dir / "learning_curve.png",
        dpi=160
    )

    plt.close(fig)

    print(
        f"Best epoch: {best_epoch} | "
        f"Train macro-F1: {train_scores['macro_f1']:.4f} | "
        f"Dev macro-F1: {dev_scores['macro_f1']:.4f}"
    )

    print("Saved:", results_file)
    print("Official test: NOT ACCESSED")


if __name__ == "__main__":
    main()
