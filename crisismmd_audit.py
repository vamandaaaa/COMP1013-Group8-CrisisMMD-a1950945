"""Audit CrisisMMD splits, cached features, and duplicate candidates."""

import argparse
import csv
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np


SPLITS = ("train", "dev", "test")


def read_csv(archive, filename, delimiter=","):
    with archive.open(filename) as source:
        reader = csv.DictReader(
            io.TextIOWrapper(source, encoding="utf-8-sig"),
            delimiter=delimiter
        )
        return list(reader)


def check_features(archive, filename, id_column, required_ids):
    path = "crisismmd_group_share/cache/" + filename

    with archive.open(path) as source:
        # Only load trusted cache files from the project group.
        with np.load(io.BytesIO(source.read()), allow_pickle=True) as cache:
            ids = set(str(value) for value in cache[id_column])
            features = cache["feat"]

            return {
                "shape": list(features.shape),
                "missing_ids": len(required_ids - ids),
                "nonfinite_rows": int(
                    (~np.isfinite(features).all(axis=1)).sum()
                ),
                "zero_rows": int(
                    (np.count_nonzero(features, axis=1) == 0).sum()
                )
            }


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="CrisisMMD data audit"
    )
    parser.add_argument("--group-zip", type=Path, required=True)
    parser.add_argument("--splits-zip", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=Path("audit_output")
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    with ZipFile(args.splits_zip) as splits_zip, \
         ZipFile(args.group_zip) as group_zip:

        # Load the original train, dev, and test splits.
        rows = {}

        for split in SPLITS:
            filename = (
                "splits/crisismmd_datasplit_agreed_label/"
                f"task_humanitarian_text_img_agreed_lab_{split}.tsv"
            )
            rows[split] = read_csv(
                splits_zip, filename, delimiter="\t"
            )

        all_rows = [
            row for split in SPLITS for row in rows[split]
        ]

        image_ids = {row["image_id"] for row in all_rows}
        tweet_ids = {row["tweet_id"] for row in all_rows}

        split_counts = {
            split: len(rows[split]) for split in SPLITS
        }

        class_counts = {
            split: dict(Counter(
                row["label"] for row in rows[split]
            ))
            for split in SPLITS
        }

        lookups = {
            split: {
                row["image_id"]: row for row in rows[split]
            }
            for split in SPLITS
        }

        # Check cross-split pHash candidates.
        pairs = read_csv(
            group_zip,
            "crisismmd_group_share/cache/"
            "near_duplicates_humanitarian.csv"
        )

        pair_rows = []
        by_split = Counter()
        conflict_count = 0

        flagged = {
            cutoff: defaultdict(set)
            for cutoff in (0, 2, 4, 6)
        }

        for pair in pairs:
            a = pair["split_A"]
            b = pair["split_B"]

            image_a = pair["image_id_A"]
            image_b = pair["image_id_B"]

            record_a = lookups[a].get(image_a)
            record_b = lookups[b].get(image_b)

            distance = int(pair["phash_distance"])

            label_a = (
                record_a["label"] if record_a else "NOT_FOUND"
            )
            label_b = (
                record_b["label"] if record_b else "NOT_FOUND"
            )

            conflict = bool(
                record_a and record_b and label_a != label_b
            )

            conflict_count += conflict
            by_split[f"{a}-{b}"] += 1

            pair_rows.append({
                "split_A": a,
                "split_B": b,
                "image_id_A": image_a,
                "image_id_B": image_b,
                "event_A": (
                    record_a["event_name"]
                    if record_a else "NOT_FOUND"
                ),
                "event_B": (
                    record_b["event_name"]
                    if record_b else "NOT_FOUND"
                ),
                "label_A": label_a,
                "label_B": label_b,
                "phash_distance": distance,
                "label_conflict": int(conflict)
            })

            for cutoff in flagged:
                if distance <= cutoff:
                    flagged[cutoff][a].add(image_a)
                    flagged[cutoff][b].add(image_b)

        # Identify repeated EfficientNet feature vectors.
        with group_zip.open(
            "crisismmd_group_share/cache/image_b0.npz"
        ) as source:

            with np.load(
                io.BytesIO(source.read()),
                allow_pickle=True
            ) as cache:

                groups = defaultdict(list)

                for image_id, feature in zip(
                    cache["image_id"], cache["feat"]
                ):
                    groups[feature.tobytes()].append(str(image_id))

        repeated = sorted(
            (ids for ids in groups.values() if len(ids) > 1),
            key=len,
            reverse=True
        )

        image_lookup = {
            row["image_id"]: (split, row["label"])
            for split in SPLITS
            for row in rows[split]
        }

        feature_rows = []
        coverage = []

        for group_number, members in enumerate(repeated, start=1):
            if group_number <= 3:
                covered = [
                    image_id for image_id in members
                    if image_id in image_lookup
                ]

                coverage.append({
                    "group": group_number,
                    "total_cached_images": len(members),
                    "agreed_task_images": len(covered)
                })

            for image_id in members:
                split, label = image_lookup.get(
                    image_id,
                    ("outside_agreed_humanitarian", "")
                )

                feature_rows.append({
                    "feature_group": group_number,
                    "group_size": len(members),
                    "image_id": image_id,
                    "split": split,
                    "humanitarian_label": label
                })

        # Check whether all required frozen features exist.
        features = {
            "bert": check_features(
                group_zip,
                "text_bert.npz",
                "tweet_id",
                tweet_ids
            ),
            "efficientnet_b0": check_features(
                group_zip,
                "image_b0.npz",
                "image_id",
                image_ids
            )
        }

    # Save candidate pairs and repeated feature groups.
    write_csv(
        args.out / "phash_candidates_for_review.csv",
        pair_rows,
        [
            "split_A", "split_B",
            "image_id_A", "image_id_B",
            "event_A", "event_B",
            "label_A", "label_B",
            "phash_distance", "label_conflict"
        ]
    )

    write_csv(
        args.out / "identical_image_feature_groups.csv",
        feature_rows,
        [
            "feature_group", "group_size", "image_id",
            "split", "humanitarian_label"
        ]
    )

    summary = {
        "total_rows": len(all_rows),
        "unique_tweet_ids": len(tweet_ids),
        "unique_image_ids": len(image_ids),
        "split_counts": split_counts,
        "class_counts": class_counts,
        "feature_caches": features,
        "phash_pair_count": len(pairs),
        "phash_pairs_by_split": dict(by_split),
        "phash_conflicting_labels": conflict_count,
        "phash_flagged_images": {
            str(cutoff): {
                split: len(flagged[cutoff][split])
                for split in SPLITS
            }
            for cutoff in flagged
        },
        "repeated_feature_groups": len(repeated),
        "largest_feature_groups": [
            len(group) for group in repeated[:3]
        ],
        "largest_group_coverage": coverage,
        "note": (
            "Candidate matches need visual review; "
            "no data was deleted."
        )
    }

    (args.out / "audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8"
    )

    print("CrisisMMD — Dataset Audit")
    print("=" * 35)

    for split in SPLITS:
        print(f"{split}: {split_counts[split]} samples")

    print("Total:", len(all_rows))
    print("pHash candidate pairs:", len(pairs))
    print("Conflicting-label pairs:", conflict_count)
    print("Identical feature groups:", len(repeated))
    print("Largest groups:", [len(g) for g in repeated[:3]])
    print(
        "Missing cached IDs:",
        {
            name: value["missing_ids"]
            for name, value in features.items()
        }
    )
    print("Reports saved in:", args.out)


if __name__ == "__main__":
    main()
