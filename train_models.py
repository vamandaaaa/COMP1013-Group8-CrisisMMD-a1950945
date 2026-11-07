"""Train CrisisMMD unimodal classifiers and multimodal fusion models.

Uses cleaned train and development splits with frozen BERT
and EfficientNet-B0 features. The test set is not used.
"""

import argparse
import copy
import csv
import io
import json
import random
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def read_split(path):
    """Read a cleaned TSV split."""
    with open(path, encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def read_features(archive, filename, id_column):
    """Load frozen features from the trusted group cache."""
    cache_path = "crisismmd_group_share/cache/" + filename

    with archive.open(cache_path) as source:
        with np.load(
            io.BytesIO(source.read()),
            allow_pickle=True
        ) as cache:
            ids = [str(value) for value in cache[id_column]]
            features = cache["feat"].astype(np.float32)

    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate feature IDs: " + filename)

    lookup = {
        image_id: index
        for index, image_id in enumerate(ids)
    }

    return lookup, features


def features_for_rows(rows, id_column, lookup, features):
    """Align cached features with dataset rows."""
    missing = [
        row[id_column]
        for row in rows
        if row[id_column] not in lookup
    ]

    if missing:
        raise ValueError(
            f"{len(missing)} missing {id_column} values"
        )

    indices = [
        lookup[row[id_column]]
        for row in rows
    ]

    return features[indices]


class Classifier(nn.Module):
    """Small feed-forward classifier for frozen features."""

    def __init__(
        self,
        input_size,
        n_classes,
        hidden=128,
        dropout=0.3
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, x):
        return self.layers(x)


def predict_probabilities(model, features, device):
    """Return class probabilities without updating the model."""
    model.eval()

    with torch.no_grad():
        inputs = torch.from_numpy(features).to(device)
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1)

    return probabilities.cpu().numpy()


def calculate_scores(y_true, probabilities, n_classes):
    """Calculate accuracy and F1 scores."""
    predictions = probabilities.argmax(axis=1)
    labels = list(range(n_classes))

    return {
        "accuracy": round(
            float(accuracy_score(y_true, predictions)), 4
        ),
        "macro_f1": round(
            float(f1_score(
                y_true,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0
            )), 4
        ),
        "weighted_f1": round(
            float(f1_score(
                y_true,
                predictions,
                labels=labels,
                average="weighted",
                zero_division=0
            )), 4
        )
    }


def train_model(
    name,
    x_train,
    y_train,
    x_dev,
    y_dev,
    n_classes,
    class_weights,
    args,
    device
):
    """Train one classifier with validation-based early stopping."""

    model = Classifier(
        input_size=x_train.shape[1],
        n_classes=n_classes,
        hidden=args.hidden,
        dropout=args.dropout
    ).to(device)

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=device
        )
    )

    dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train)
    )

    generator = torch.Generator().manual_seed(args.seed)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator
    )

    x_tr = torch.from_numpy(x_train).to(device)
    x_va = torch.from_numpy(x_dev).to(device)

    y_tr = torch.from_numpy(y_train).to(device)
    y_va = torch.from_numpy(y_dev).to(device)

    best_f1 = -1.0
    best_loss = float("inf")
    best_epoch = 0
    best_state = None

    history = []
    stagnant_epochs = 0

    for epoch in range(1, args.epochs + 1):

        # Training
        model.train()

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimiser.zero_grad()

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimiser.step()

        # Evaluation at the end of each epoch
        model.eval()

        with torch.no_grad():
            train_logits = model(x_tr)
            dev_logits = model(x_va)

            train_loss = float(
                criterion(train_logits, y_tr).item()
            )
            dev_loss = float(
                criterion(dev_logits, y_va).item()
            )

            train_probs = torch.softmax(
                train_logits, dim=1
            ).cpu().numpy()

            dev_probs = torch.softmax(
                dev_logits, dim=1
            ).cpu().numpy()

        labels = list(range(n_classes))

        train_f1 = f1_score(
            y_train,
            train_probs.argmax(axis=1),
            labels=labels,
            average="macro",
            zero_division=0
        )

        dev_f1 = f1_score(
            y_dev,
            dev_probs.argmax(axis=1),
            labels=labels,
            average="macro",
            zero_division=0
        )

        history.append({
            "model": name,
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "dev_loss": round(dev_loss, 5),
            "train_macro_f1": round(float(train_f1), 5),
            "dev_macro_f1": round(float(dev_f1), 5)
        })

        print(
            f"{name:12} epoch {epoch:02d} | "
            f"train F1 {train_f1:.3f} | "
            f"dev F1 {dev_f1:.3f}"
        )

        # Select the best checkpoint using development macro-F1.
        improved = (
            dev_f1 > best_f1 + 1e-6
            or (
                abs(dev_f1 - best_f1) <= 1e-6
                and dev_loss < best_loss
            )
        )

        if improved:
            best_f1 = float(dev_f1)
            best_loss = dev_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stagnant_epochs = 0
        else:
            stagnant_epochs += 1

            if stagnant_epochs >= args.patience:
                break

    # Restore the selected checkpoint.
    model.load_state_dict(best_state)

    train_probs = predict_probabilities(
        model, x_train, device
    )
    dev_probs = predict_probabilities(
        model, x_dev, device
    )

    metrics = {
        "best_epoch": best_epoch,
        "train": calculate_scores(
            y_train, train_probs, n_classes
        ),
        "dev": calculate_scores(
            y_dev, dev_probs, n_classes
        )
    }

    return model, metrics, history, dev_probs


def plot_curves(history, output_path):
    """Save training-loss and validation-F1 plots."""

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    names = sorted({
        row["model"] for row in history
    })

    for name in names:
        rows = [
            row for row in history
            if row["model"] == name
        ]

        epochs = [row["epoch"] for row in rows]

        axes[0].plot(
            epochs,
            [row["train_loss"] for row in rows],
            label=name + " train"
        )

        axes[0].plot(
            epochs,
            [row["dev_loss"] for row in rows],
            linestyle="--",
            label=name + " dev"
        )

        axes[1].plot(
            epochs,
            [row["dev_macro_f1"] for row in rows],
            label=name
        )

    axes[0].set(
        xlabel="Epoch",
        ylabel="Weighted cross-entropy",
        title="Training and validation loss"
    )

    axes[1].set(
        xlabel="Epoch",
        ylabel="Macro-F1",
        title="Validation macro-F1"
    )

    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=8)

    fig.tight_layout()

    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def add_detailed_metrics(metrics, y_true, probabilities, labels):
    """Add per-class results and a confusion matrix."""

    predictions = probabilities.argmax(axis=1)
    class_ids = list(range(len(labels)))

    metrics["per_class"] = classification_report(
        y_true,
        predictions,
        labels=class_ids,
        target_names=labels,
        output_dict=True,
        zero_division=0
    )

    metrics["confusion_matrix"] = confusion_matrix(
        y_true,
        predictions,
        labels=class_ids
    ).tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Train CrisisMMD unimodal and fusion classifiers"
    )

    parser.add_argument(
        "--group-zip",
        type=Path,
        default=Path("crisismmd_group_share.zip")
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed")
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/experiments")
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models")
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures")
    )

    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["cpu", "mps"],
        default="cpu"
    )

    args = parser.parse_args()

    # Fix random seeds for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "mps":
        if not torch.backends.mps.is_available():
            parser.error("MPS unavailable; use --device cpu")

    device = torch.device(args.device)

    # Read train and dev only.
    train = read_split(
        args.processed_dir / "humanitarian_train.tsv"
    )
    dev = read_split(
        args.processed_dir / "humanitarian_dev.tsv"
    )

    if not train or not dev:
        parser.error("Cleaned train/dev splits must be nonempty")

    for column in ("tweet_id", "image_id"):
        train_ids = {row[column] for row in train}
        dev_ids = {row[column] for row in dev}

        if train_ids & dev_ids:
            raise ValueError(
                f"{column} overlaps between train and dev"
            )

    labels = sorted({
        row["label"] for row in train
    })

    if (
        len(labels) != 5
        or {row["label"] for row in dev} != set(labels)
    ):
        raise ValueError(
            "Expected five humanitarian labels in train and dev"
        )

    label_to_int = {
        label: index
        for index, label in enumerate(labels)
    }

    y_train = np.array(
        [label_to_int[row["label"]] for row in train],
        dtype=np.int64
    )

    y_dev = np.array(
        [label_to_int[row["label"]] for row in dev],
        dtype=np.int64
    )

    # Load pretrained BERT and EfficientNet feature vectors.
    with ZipFile(args.group_zip) as archive:
        text_lookup, text_features = read_features(
            archive,
            "text_bert.npz",
            "tweet_id"
        )

        image_lookup, image_features = read_features(
            archive,
            "image_b0.npz",
            "image_id"
        )

    xt_train = features_for_rows(
        train, "tweet_id", text_lookup, text_features
    )
    xt_dev = features_for_rows(
        dev, "tweet_id", text_lookup, text_features
    )

    xi_train = features_for_rows(
        train, "image_id", image_lookup, image_features
    )
    xi_dev = features_for_rows(
        dev, "image_id", image_lookup, image_features
    )

    # Fit scalers on training data only.
    text_scaler = StandardScaler()
    image_scaler = StandardScaler()

    xt_train = text_scaler.fit_transform(
        xt_train
    ).astype(np.float32)

    xt_dev = text_scaler.transform(
        xt_dev
    ).astype(np.float32)

    xi_train = image_scaler.fit_transform(
        xi_train
    ).astype(np.float32)

    xi_dev = image_scaler.transform(
        xi_dev
    ).astype(np.float32)

    # Early fusion concatenates text and image features.
    xe_train = np.concatenate(
        [xt_train, xi_train], axis=1
    )

    xe_dev = np.concatenate(
        [xt_dev, xi_dev], axis=1
    )

    # Smoothed inverse-frequency class weights.
    counts = np.bincount(
        y_train, minlength=len(labels)
    )

    class_weights = np.sqrt(
        len(y_train) / (len(labels) * counts)
    )

    class_weights /= class_weights.mean()

    print("CrisisMMD — Model Training")
    print("=" * 45)
    print("Train:", len(train))
    print("Dev:", len(dev))
    print("Test: not accessed")
    print("Classes:", dict(zip(labels, counts.tolist())))
    print("Device:", device)
    print("Dropout:", args.dropout)
    print()

    args.results_dir.mkdir(
        parents=True, exist_ok=True
    )

    args.models_dir.mkdir(
        parents=True, exist_ok=True
    )

    # Three independently trained classifiers.
    model_data = {
        "text_only": (xt_train, xt_dev),
        "image_only": (xi_train, xi_dev),
        "early_fusion": (xe_train, xe_dev)
    }

    results = {}
    all_history = []
    dev_probabilities = {}

    for name, (x_train, x_dev) in model_data.items():

        model, metrics, history, probabilities = train_model(
            name,
            x_train,
            y_train,
            x_dev,
            y_dev,
            len(labels),
            class_weights,
            args,
            device
        )

        add_detailed_metrics(
            metrics["dev"],
            y_dev,
            probabilities,
            labels
        )

        results[name] = metrics
        all_history.extend(history)
        dev_probabilities[name] = probabilities

        torch.save(
            model.state_dict(),
            args.models_dir / f"{name}.pt"
        )

        print(
            f"{name}: best epoch {metrics['best_epoch']}, "
            f"dev macro-F1 {metrics['dev']['macro_f1']:.4f}"
        )
        print()

    # Late fusion combines unimodal prediction probabilities.
    # The mixing weight is selected on dev only.
    best_alpha = None
    best_score = -1.0

    for alpha in np.linspace(0, 1, 21):

        combined = (
            alpha * dev_probabilities["text_only"]
            + (1.0 - alpha) * dev_probabilities["image_only"]
        )

        score = f1_score(
            y_dev,
            combined.argmax(axis=1),
            labels=list(range(len(labels))),
            average="macro",
            zero_division=0
        )

        if score > best_score:
            best_score = float(score)
            best_alpha = float(alpha)

    late_probabilities = (
        best_alpha * dev_probabilities["text_only"]
        + (1.0 - best_alpha) * dev_probabilities["image_only"]
    )

    late_metrics = calculate_scores(
        y_dev,
        late_probabilities,
        len(labels)
    )

    add_detailed_metrics(
        late_metrics,
        y_dev,
        late_probabilities,
        labels
    )

    results["late_fusion"] = {
        "text_probability_weight": round(best_alpha, 2),
        "dev": late_metrics,
        "note": (
            "Weight selected on dev; "
            "not a separately trained neural network"
        )
    }

    # Save training history for the learning curves.
    history_path = (
        args.results_dir / "training_history.csv"
    )

    with history_path.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(all_history[0].keys()),
            lineterminator="\n"
        )

        writer.writeheader()
        writer.writerows(all_history)

    figure_path = (
        args.figures_dir / "training_curves.png"
    )

    plot_curves(all_history, figure_path)

    # Save scalers for inference on the final test data.
    np.savez_compressed(
        args.models_dir / "feature_scalers.npz",
        text_mean=text_scaler.mean_,
        text_scale=text_scaler.scale_,
        image_mean=image_scaler.mean_,
        image_scale=image_scaler.scale_
    )

    # Save the full development evaluation.
    config = {
        "seed": args.seed,
        "device": str(device),
        "epochs_max": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "learning_rate": args.lr,
        "labels": labels,
        "class_weights": class_weights.round(4).tolist(),
        "split_sizes": {
            "train": len(train),
            "dev": len(dev)
        },
        "encoder_features": (
            "Frozen BERT 768D + EfficientNet-B0 1280D "
            "from group cache"
        ),
        "evaluation": "dev only; test not accessed",
        "results": results
    }

    result_path = args.results_dir / "dev_results.json"

    result_path.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8"
    )

    print("Validation results:")

    for name, result in results.items():
        print(
            f"{name:12} "
            f"macro-F1={result['dev']['macro_f1']:.4f} "
            f"accuracy={result['dev']['accuracy']:.4f}"
        )

    print()
    print("Saved:", result_path)
    print("Saved:", history_path)
    print("Saved:", figure_path)
    print("Test set was NOT accessed.")


if __name__ == "__main__":
    main()
