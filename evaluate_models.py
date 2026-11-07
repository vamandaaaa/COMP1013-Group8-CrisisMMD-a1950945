"""Evaluate tuned CrisisMMD classifiers on the test dataset.

All model configurations were selected using validation results.
This script loads saved models without retraining them.
"""

import csv
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import torch

from scipy.stats import binomtest
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from train_models import (
    Classifier,
    read_split,
    read_features,
    features_for_rows
)


# Best configurations selected from validation results.
SELECTED = {
    "text_only": "d05_lr0003",
    "image_only": "d03_lr003",
    "early_fusion": "d05_lr003",
    "late_fusion": "d05_lr001"
}

ROOT = Path(".")
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
PROCESSED = ROOT / "data/processed"
OUTPUT = RESULTS / "evaluation"
FIGURES = ROOT / "figures"


def read_configs():
    """Read the saved validation configuration for each experiment."""
    configs = {}

    for experiment in set(SELECTED.values()):
        path = (
            RESULTS / "experiments" /
            experiment / "dev_results.json"
        )

        with open(path, encoding="utf-8") as file:
            configs[experiment] = json.load(file)

    return configs


def check_data(train, dev, test, configs):
    """Check labels, split sizes, and ID overlap."""
    labels = configs[SELECTED["early_fusion"]]["labels"]

    if len(labels) != 5:
        raise ValueError("Expected five humanitarian classes")

    expected_lr = {
        "d05_lr0003": 0.0003,
        "d03_lr003": 0.003,
        "d05_lr003": 0.003,
        "d05_lr001": 0.001
    }

    for experiment, config in configs.items():

        if config["labels"] != labels:
            raise ValueError(
                f"Class order mismatch: {experiment}"
            )

        expected_sizes = {
            "train": len(train),
            "dev": len(dev)
        }

        if config["split_sizes"] != expected_sizes:
            raise ValueError(
                f"Split sizes do not match: {experiment}"
            )

        dropout = (
            0.5 if experiment.startswith("d05") else 0.3
        )

        if not np.isclose(config["dropout"], dropout):
            raise ValueError(
                f"Unexpected dropout: {experiment}"
            )

        if not np.isclose(
            config["learning_rate"],
            expected_lr[experiment]
        ):
            raise ValueError(
                f"Unexpected learning rate: {experiment}"
            )

    # Check for overlapping tweet and image IDs.
    for column in ("tweet_id", "image_id"):

        ids = [
            {row[column] for row in rows}
            for rows in (train, dev, test)
        ]

        # A tweet may have multiple images within one split.
        # Image IDs must still be unique.
        if column == "image_id" and (
            len(ids[0]) != len(train)
            or len(ids[1]) != len(dev)
            or len(ids[2]) != len(test)
        ):
            raise ValueError(
                f"Repeated {column} within a split"
            )

        if (
            ids[0] & ids[1]
            or ids[0] & ids[2]
            or ids[1] & ids[2]
        ):
            raise ValueError(
                f"Overlapping {column} between splits"
            )

    if any(row["label"] not in labels for row in test):
        raise ValueError("Unexpected test label")

    return labels


def get_features(test):
    """Load and align the frozen feature vectors."""

    with ZipFile("crisismmd_group_share.zip") as archive:

        text_index, text_all = read_features(
            archive,
            "text_bert.npz",
            "tweet_id"
        )

        image_index, image_all = read_features(
            archive,
            "image_b0.npz",
            "image_id"
        )

    text = features_for_rows(
        test, "tweet_id", text_index, text_all
    )

    images = features_for_rows(
        test, "image_id", image_index, image_all
    )

    return text, images


def predict_checkpoint(
    experiment,
    model_name,
    text,
    images,
    configs,
    n_classes
):
    """Load a checkpoint and return its class probabilities."""

    config = configs[experiment]
    checkpoint_dir = MODELS / experiment

    # Use the scalers fitted during training.
    with np.load(
        checkpoint_dir / "feature_scalers.npz",
        allow_pickle=False
    ) as scalers:

        text_scaled = (
            (text - scalers["text_mean"])
            / scalers["text_scale"]
        ).astype(np.float32)

        image_scaled = (
            (images - scalers["image_mean"])
            / scalers["image_scale"]
        ).astype(np.float32)

    if model_name == "text_only":
        x = text_scaled

    elif model_name == "image_only":
        x = image_scaled

    elif model_name == "early_fusion":
        x = np.concatenate(
            [text_scaled, image_scaled],
            axis=1
        )

    else:
        raise ValueError("Unknown model: " + model_name)

    # Recreate the same classifier architecture.
    model = Classifier(
        input_size=x.shape[1],
        n_classes=n_classes,
        hidden=config["hidden"],
        dropout=config["dropout"]
    )

    weights = torch.load(
        checkpoint_dir / f"{model_name}.pt",
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(weights)
    model.eval()

    with torch.no_grad():
        inputs = torch.from_numpy(x)
        logits = model(inputs)

        probabilities = torch.softmax(
            logits, dim=1
        ).numpy()

    return probabilities


def calculate_metrics(actual, predicted, labels):
    """Calculate overall and per-class evaluation metrics."""

    class_ids = list(range(len(labels)))

    report = classification_report(
        actual,
        predicted,
        labels=class_ids,
        target_names=labels,
        output_dict=True,
        zero_division=0
    )

    per_class = {}

    for name in labels:
        per_class[name] = {}

        for key, value in report[name].items():
            if key == "support":
                per_class[name][key] = int(value)
            else:
                per_class[name][key] = round(
                    float(value), 5
                )

    return {
        "accuracy": round(
            float(accuracy_score(actual, predicted)), 5
        ),
        "macro_f1": round(
            float(f1_score(
                actual,
                predicted,
                labels=class_ids,
                average="macro",
                zero_division=0
            )), 5
        ),
        "weighted_f1": round(
            float(f1_score(
                actual,
                predicted,
                labels=class_ids,
                average="weighted",
                zero_division=0
            )), 5
        ),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            actual,
            predicted,
            labels=class_ids
        ).tolist()
    }


def bootstrap_results(actual, predictions, n_classes):
    """Calculate stratified paired bootstrap confidence intervals."""

    rng = np.random.default_rng(123)
    repetitions = 2000

    class_ids = list(range(n_classes))

    groups = [
        np.where(actual == class_id)[0]
        for class_id in class_ids
    ]

    scores = {
        name: [] for name in predictions
    }

    comparisons = [
        ("early_fusion", "text_only"),
        ("early_fusion", "image_only"),
        ("late_fusion", "text_only"),
        ("late_fusion", "image_only"),
        ("early_fusion", "late_fusion")
    ]

    differences = {}

    for a, b in comparisons:
        differences[f"{a}_minus_{b}"] = []

    for _ in range(repetitions):

        # Sample each class separately, preserving its size.
        indexes = np.concatenate([
            rng.choice(
                group,
                size=len(group),
                replace=True
            )
            for group in groups
        ])

        sampled_true = actual[indexes]
        sampled_scores = {}

        for name, predicted in predictions.items():

            score = f1_score(
                sampled_true,
                predicted[indexes],
                labels=class_ids,
                average="macro",
                zero_division=0
            )

            scores[name].append(score)
            sampled_scores[name] = score

        for a, b in comparisons:
            key = f"{a}_minus_{b}"

            differences[key].append(
                sampled_scores[a] - sampled_scores[b]
            )

    def interval(values):
        bounds = np.quantile(
            values, [0.025, 0.975]
        )
        return [
            round(float(value), 5)
            for value in bounds
        ]

    observed = {}

    for name, predicted in predictions.items():
        observed[name] = f1_score(
            actual,
            predicted,
            labels=class_ids,
            average="macro",
            zero_division=0
        )

    paired_results = {}

    for a, b in comparisons:
        key = f"{a}_minus_{b}"
        bounds = interval(differences[key])

        paired_results[key] = {
            "point_estimate": round(
                float(observed[a] - observed[b]), 5
            ),
            "ci95": bounds,
            "ci_excludes_zero": (
                bounds[0] > 0 or bounds[1] < 0
            )
        }

    return {
        "method": "Class-stratified paired bootstrap",
        "repetitions": repetitions,
        "model_macro_f1_95ci": {
            name: interval(values)
            for name, values in scores.items()
        },
        "paired_macro_f1_difference": paired_results,
        "limitation": (
            "Does not account for event-level or "
            "near-duplicate clustering."
        )
    }


def mcnemar_test(actual, prediction_a, prediction_b):
    """Compare paired classification accuracy errors."""

    a_correct = prediction_a == actual
    b_correct = prediction_b == actual

    a_only = int(np.sum(
        a_correct & ~b_correct
    ))

    b_only = int(np.sum(
        ~a_correct & b_correct
    ))

    total = a_only + b_only

    if total:
        p_value = float(
            binomtest(a_only, total, p=0.5).pvalue
        )
    else:
        p_value = 1.0

    return {
        "a_correct_b_wrong": a_only,
        "a_wrong_b_correct": b_only,
        "p_value_unadjusted": p_value,
        "note": (
            "McNemar tests paired accuracy, "
            "not macro-F1."
        )
    }


def event_analysis(test, actual, predictions, labels):
    """Evaluate performance separately for each disaster event."""

    results = {}

    events = sorted({
        row["event_name"] for row in test
    })

    for event in events:

        indexes = np.array([
            i for i, row in enumerate(test)
            if row["event_name"] == event
        ])

        truth = actual[indexes]
        present_classes = sorted(set(truth.tolist()))

        model_results = {}

        for name, predicted in predictions.items():

            event_predictions = predicted[indexes]

            model_results[name] = {
                "accuracy": round(
                    float(accuracy_score(
                        truth, event_predictions
                    )), 5
                ),
                "macro_f1_present_classes": round(
                    float(f1_score(
                        truth,
                        event_predictions,
                        labels=present_classes,
                        average="macro",
                        zero_division=0
                    )), 5
                )
            }

        results[event] = {
            "samples": len(indexes),
            "classes_present": [
                labels[i] for i in present_classes
            ],
            "models": model_results
        }

    return results


def near_duplicate_analysis(
    train,
    dev,
    test,
    actual,
    predictions,
    labels
):
    """Check sensitivity to unresolved cross-split pHash pairs."""

    path = (
        PROCESSED / "unresolved_phash_pairs.csv"
    )

    with open(path, newline="", encoding="utf-8") as file:
        pairs = list(csv.DictReader(file))

    other_ids = {
        row["image_id"]
        for row in train + dev
    }

    test_ids = {
        row["image_id"]
        for row in test
    }

    flagged = set()

    for pair in pairs:
        a = pair["image_id_A"]
        b = pair["image_id_B"]

        if a in test_ids and b in other_ids:
            flagged.add(a)

        if b in test_ids and a in other_ids:
            flagged.add(b)

    remaining = np.array([
        i for i, row in enumerate(test)
        if row["image_id"] not in flagged
    ])

    subset_results = {}

    if len(remaining) > 0:

        subset_true = actual[remaining]

        class_support = {}

        for i, label in enumerate(labels):
            class_support[label] = int(
                np.sum(subset_true == i)
            )

        subset_scores = {}

        for name, predicted in predictions.items():

            subset_scores[name] = round(
                float(f1_score(
                    subset_true,
                    predicted[remaining],
                    labels=list(range(len(labels))),
                    average="macro",
                    zero_division=0
                )), 5
            )

        subset_results = {
            "class_support": class_support,
            "macro_f1": subset_scores
        }

    result = {
        "candidate_pairs": len(pairs),
        "flagged_test_images": len(flagged),
        "remaining_test_samples": len(remaining),
        "subset_sensitivity": subset_results,
        "limitation": (
            "Exploratory check, not proof "
            "of zero visual leakage."
        )
    }

    return result, flagged


def save_confusion_matrices(actual, predictions, labels):
    """Plot confusion matrices for all four models."""

    names = [
        "text_only",
        "image_only",
        "early_fusion",
        "late_fusion"
    ]

    short_labels = [
        "Affected",
        "Damage",
        "Not human.",
        "Other",
        "Rescue"
    ]

    fig, axes = plt.subplots(
        2, 2, figsize=(11, 9)
    )

    for axis, name in zip(axes.flat, names):

        matrix = confusion_matrix(
            actual,
            predictions[name],
            labels=list(range(len(labels)))
        )

        ConfusionMatrixDisplay(
            matrix,
            display_labels=short_labels
        ).plot(
            ax=axis,
            colorbar=False,
            cmap="Blues",
            values_format="d"
        )

        axis.set_title(
            name.replace("_", " ").title()
        )

        axis.tick_params(
            axis="x",
            labelrotation=40,
            labelsize=8
        )

        axis.tick_params(
            axis="y",
            labelsize=8
        )

    fig.tight_layout()

    FIGURES.mkdir(
        parents=True,
        exist_ok=True
    )

    path = (
        FIGURES / "test_confusion_matrices.png"
    )

    fig.savefig(path, dpi=160)
    plt.close(fig)

    return path


def main():

    # Read saved configurations and the cleaned dataset.
    configs = read_configs()

    train = read_split(
        PROCESSED / "humanitarian_train.tsv"
    )

    dev = read_split(
        PROCESSED / "humanitarian_dev.tsv"
    )

    test = read_split(
        PROCESSED / "humanitarian_test.tsv"
    )

    labels = check_data(
        train, dev, test, configs
    )

    actual = np.array([
        labels.index(row["label"])
        for row in test
    ])

    if set(actual) != set(range(5)):
        raise ValueError(
            "Test set must contain all five classes"
        )

    # Load the test text and image features.
    text, images = get_features(test)

    probabilities = {}

    # Evaluate the individually selected model checkpoints.
    for name in (
        "text_only",
        "image_only",
        "early_fusion"
    ):

        experiment = SELECTED[name]

        probabilities[name] = predict_checkpoint(
            experiment,
            name,
            text,
            images,
            configs,
            len(labels)
        )

    # Late fusion uses BOTH unimodal checkpoints from E4.
    late_experiment = SELECTED["late_fusion"]

    late_text = predict_checkpoint(
        late_experiment,
        "text_only",
        text,
        images,
        configs,
        len(labels)
    )

    late_image = predict_checkpoint(
        late_experiment,
        "image_only",
        text,
        images,
        configs,
        len(labels)
    )

    alpha = float(
        configs[late_experiment]["results"]
        ["late_fusion"]["text_probability_weight"]
    )

    probabilities["late_fusion"] = (
        alpha * late_text
        + (1.0 - alpha) * late_image
    )

    predictions = {
        name: probs.argmax(axis=1)
        for name, probs in probabilities.items()
    }

    # Overall metrics and confusion matrices.
    model_results = {}

    for name, predicted in predictions.items():
        model_results[name] = calculate_metrics(
            actual,
            predicted,
            labels
        )

    # Statistical comparison.
    bootstrap = bootstrap_results(
        actual,
        predictions,
        len(labels)
    )

    comparisons = {
        "early_vs_text": mcnemar_test(
            actual,
            predictions["early_fusion"],
            predictions["text_only"]
        ),
        "early_vs_late": mcnemar_test(
            actual,
            predictions["early_fusion"],
            predictions["late_fusion"]
        ),
        "late_vs_text": mcnemar_test(
            actual,
            predictions["late_fusion"],
            predictions["text_only"]
        )
    }

    sensitivity, flagged = near_duplicate_analysis(
        train,
        dev,
        test,
        actual,
        predictions,
        labels
    )

    # Save aggregate evaluation results.
    results = {
        "test_samples": len(test),
        "selected_from_dev": SELECTED,
        "late_fusion_E4_text_weight": alpha,
        "labels": labels,
        "model_results": model_results,
        "bootstrap_macro_f1": bootstrap,
        "mcnemar_accuracy": comparisons,
        "by_event": event_analysis(
            test, actual, predictions, labels
        ),
        "near_duplicate_sensitivity": sensitivity,
        "limitations": [
            "The test set was previously inspected "
            "during an earlier project run.",
            "Affected individuals has only nine "
            "test examples.",
            "Within-event evaluation is not "
            "unseen-event generalisation.",
            "Near-duplicate candidates may remain "
            "across splits."
        ]
    }

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT / "test_results.json"
    )

    output_path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8"
    )

    # Keep individual predictions local, under data/.
    prediction_dir = (
        ROOT / "data/evaluation"
    )

    prediction_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    prediction_path = (
        prediction_dir / "test_predictions.csv"
    )

    with open(
        prediction_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "tweet_id",
                "image_id",
                "event_name",
                "true_label"
            ]
            + [
                f"pred_{name}"
                for name in predictions
            ]
            + ["flagged_near_duplicate"]
        )

        for i, row in enumerate(test):

            writer.writerow([
                row["tweet_id"],
                row["image_id"],
                row["event_name"],
                row["label"],
                *[
                    labels[predictions[name][i]]
                    for name in predictions
                ],
                int(row["image_id"] in flagged)
            ])

    figure_path = save_confusion_matrices(
        actual,
        predictions,
        labels
    )

    # Print a summary.
    print("CrisisMMD — Test Evaluation")
    print("=" * 45)
    print("Test samples:", len(test))
    print()

    for name, score in model_results.items():

        print(
            f"{name:12} "
            f"macro-F1={score['macro_f1']:.4f} "
            f"accuracy={score['accuracy']:.4f}"
        )

    print()
    print(
        "Potential near-duplicate test images:",
        len(flagged)
    )
    print("Results:", output_path)
    print("Confusion matrices:", figure_path)
    print("Local predictions:", prediction_path)
    print("No training or tuning was performed.")


if __name__ == "__main__":
    main()
