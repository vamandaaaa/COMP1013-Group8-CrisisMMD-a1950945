"""Prepare cleaned CrisisMMD humanitarian classification splits."""

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageOps


SPLITS = ("train", "dev", "test")

SPLIT_PRIORITY = {
    "train": 0,
    "dev": 1,
    "test": 2
}

TASK_FILE = (
    "splits/crisismmd_datasplit_agreed_label/"
    "task_humanitarian_text_img_agreed_lab_{}.tsv"
)

# These feature groups were confirmed as placeholders by visual review.
PLACEHOLDER_GROUPS = {
    1: 33,
    2: 30,
    3: 8
}

EXPECTED_PLACEHOLDER_ROWS = {
    1: 9,
    2: 6,
    3: 3
}


def read_tsv(archive, filename):
    """Read one original split from the ZIP archive."""
    with archive.open(filename) as source:
        text = io.TextIOWrapper(source, encoding="utf-8-sig")
        return list(csv.DictReader(text, delimiter="\t"))


def read_csv(filename):
    """Read a CSV created during dataset audit."""
    with open(filename, newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def save_table(filename, rows, columns, delimiter=","):
    """Write a CSV or TSV file."""
    filename.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
            delimiter=delimiter,
            extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def pixel_hash(row, images_root, hash_cache):
    """
    Calculate SHA-256 from actual decoded image pixels.

    Unlike pHash, this checks exact pixel equality, including
    image dimensions and EXIF orientation.
    """
    image_id = row["image_id"]

    if image_id in hash_cache:
        return hash_cache[image_id]

    root = images_root.resolve()
    image_path = (root / row["image"]).resolve()

    if not image_path.is_relative_to(root):
        raise ValueError("Image path is outside the data directory")

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with Image.open(image_path) as source:
        source.load()
        image = ImageOps.exif_transpose(source).convert("RGB")

        digest = hashlib.sha256()
        digest.update(
            f"{image.width}x{image.height}:".encode()
        )
        digest.update(image.tobytes())

    result = digest.hexdigest()
    hash_cache[image_id] = result

    return result


def split_statistics(data):
    """Count rows and labels in each split."""
    statistics = {}

    for split in SPLITS:
        labels = Counter(
            row["label"] for row in data[split]
        )

        statistics[split] = {
            "rows": len(data[split]),
            "labels": dict(sorted(labels.items()))
        }

    return statistics


def main():
    parser = argparse.ArgumentParser(
        description="Conservative CrisisMMD data cleaning"
    )

    parser.add_argument(
        "--splits-zip",
        type=Path,
        default=Path("splits.zip")
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("audit_output")
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("data/raw")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed")
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("results/data_quality")
    )

    args = parser.parse_args()

    # 1. Load the original, agreed-label task splits.
    with ZipFile(args.splits_zip) as archive:
        original = {}

        for split in SPLITS:
            original[split] = read_tsv(
                archive,
                TASK_FILE.format(split)
            )

    columns = list(original["train"][0].keys())

    required_columns = {
        "image_id", "tweet_id", "image", "label"
    }

    if not required_columns.issubset(columns):
        raise ValueError("Unexpected dataset columns")

    if any(not original[split] for split in SPLITS):
        raise ValueError("An original split is empty")

    # Index each image and record its original split.
    image_lookup = {}

    for split in SPLITS:
        for row in original[split]:
            image_id = row["image_id"]

            if image_id in image_lookup:
                raise ValueError(
                    f"Repeated image ID: {image_id}"
                )

            image_lookup[image_id] = (split, row)

    # 2. Identify confirmed placeholder images.
    feature_groups = read_csv(
        args.audit_dir / "identical_image_feature_groups.csv"
    )

    placeholder_ids = set()
    confirmed_groups = {}

    for group_number, expected_size in PLACEHOLDER_GROUPS.items():
        members = [
            row for row in feature_groups
            if int(row["feature_group"]) == group_number
        ]

        if len(members) != expected_size:
            raise ValueError(
                f"Unexpected size for feature group {group_number}"
            )

        agreed_ids = [
            row["image_id"]
            for row in members
            if row["image_id"] in image_lookup
        ]

        if len(agreed_ids) != EXPECTED_PLACEHOLDER_ROWS[group_number]:
            raise ValueError(
                f"Unexpected task coverage for group {group_number}"
            )

        placeholder_ids.update(agreed_ids)

        confirmed_groups[str(group_number)] = {
            "cache_size": expected_size,
            "agreed_ids": len(agreed_ids)
        }

    # 3. Examine cross-split pHash candidates.
    candidates = read_csv(
        args.audit_dir / "phash_candidates_for_review.csv"
    )

    hash_cache = {}
    exact_pairs = []
    unresolved_pairs = []
    placeholder_pairs = []

    for pair in candidates:
        image_a = pair["image_id_A"]
        image_b = pair["image_id_B"]

        if image_a not in image_lookup or image_b not in image_lookup:
            raise ValueError("Candidate image missing from original splits")

        split_a, row_a = image_lookup[image_a]
        split_b, row_b = image_lookup[image_b]

        if split_a == split_b:
            raise ValueError(
                "Expected cross-split candidate"
            )

        # Placeholders have already been marked for exclusion.
        if image_a in placeholder_ids or image_b in placeholder_ids:
            placeholder_pairs.append(pair)
            continue

        hash_a = pixel_hash(
            row_a, args.images_root, hash_cache
        )
        hash_b = pixel_hash(
            row_b, args.images_root, hash_cache
        )

        if hash_a == hash_b:
            exact_pairs.append(pair)
        else:
            unresolved_pairs.append(pair)

    # 4. Remove pixel-identical overlap between splits.
    #
    # Priority: test > dev > train.
    # This policy uses image identity and split membership,
    # not the humanitarian class labels.
    duplicate_ids = set()

    for pair in exact_pairs:
        image_a = pair["image_id_A"]
        image_b = pair["image_id_B"]

        split_a = image_lookup[image_a][0]
        split_b = image_lookup[image_b][0]

        if SPLIT_PRIORITY[split_a] < SPLIT_PRIORITY[split_b]:
            duplicate_ids.add(image_a)
        else:
            duplicate_ids.add(image_b)

    duplicate_ids -= placeholder_ids
    excluded_ids = placeholder_ids | duplicate_ids

    # Confirm that every known exact-pixel overlap was handled.
    for pair in exact_pairs:
        image_a = pair["image_id_A"]
        image_b = pair["image_id_B"]

        if image_a not in excluded_ids and image_b not in excluded_ids:
            raise AssertionError(
                "An exact-pixel cross-split duplicate remains"
            )

    # 5. Create the cleaned splits without changing labels.
    cleaned = {}

    for split in SPLITS:
        cleaned[split] = [
            row for row in original[split]
            if row["image_id"] not in excluded_ids
        ]

        save_table(
            args.out / f"humanitarian_{split}.tsv",
            cleaned[split],
            columns,
            delimiter="\t"
        )

    # 6. Save a local manifest of excluded records.
    excluded_rows = []

    for image_id in sorted(excluded_ids):
        split, row = image_lookup[image_id]

        if image_id in placeholder_ids:
            reason = "visual_verified_placeholder_group_1_to_3"
        else:
            reason = "cross_split_exact_decoded_pixels"

        excluded_rows.append({
            "image_id": image_id,
            "split": split,
            "label": row["label"],
            "reason": reason
        })

    save_table(
        args.out / "excluded_rows.csv",
        excluded_rows,
        ["image_id", "split", "label", "reason"]
    )

    save_table(
        args.out / "unresolved_phash_pairs.csv",
        unresolved_pairs,
        [
            "split_A",
            "split_B",
            "image_id_A",
            "image_id_B",
            "phash_distance",
            "label_A",
            "label_B",
            "label_conflict"
        ]
    )

    # 7. Generate a summary for the project report.
    reasons = {
        "placeholder_by_split": dict(
            Counter(
                image_lookup[image_id][0]
                for image_id in placeholder_ids
            )
        ),
        "cross_split_exact_pixels_by_split": dict(
            Counter(
                image_lookup[image_id][0]
                for image_id in duplicate_ids
            )
        )
    }

    summary = {
        "dataset": "CrisisMMD agreed-label five-class humanitarian task",
        "policy": {
            "placeholder": (
                "Exclude reviewed feature groups 1-3 only "
                "(18 agreed-task images)."
            ),
            "cross_split_exact_pixels": (
                "Exclude lower-priority split "
                "(train < dev < test); no label edits."
            ),
            "ambiguous_phash": (
                "Retained for review; pHash <= 6 alone "
                "is insufficient to confirm duplicates."
            ),
            "pixel_definition": (
                "Image decoded as RGB with EXIF orientation applied; "
                "SHA-256 of dimensions plus pixels."
            ),
            "original_files_modified": False
        },
        "verified_placeholder_groups": confirmed_groups,
        "before": split_statistics(original),
        "after": split_statistics(cleaned),
        "excluded_rows": len(excluded_ids),
        "excluded_by_reason": reasons,
        "phash_pairs_total": len(candidates),
        "phash_pairs_with_placeholder": len(placeholder_pairs),
        "cross_split_exact_pixel_pairs": len(exact_pairs),
        "unresolved_phash_pairs": len(unresolved_pairs),
        "caveat": (
            "Unresolved pHash candidates may contain near-duplicates. "
            "This is not a guarantee of zero visual leakage."
        )
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)

    report_path = args.report_dir / "cleaning_summary.json"

    report_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8"
    )

    print("CrisisMMD — Conservative Split Preparation")
    print("=" * 50)

    for split in SPLITS:
        print(
            f"{split:5} "
            f"{len(original[split]):5} -> "
            f"{len(cleaned[split]):5} rows"
        )

    print(
        "Confirmed placeholders excluded:",
        len(placeholder_ids),
        reasons["placeholder_by_split"]
    )
    print(
        "Cross-split exact-pixel pairs:",
        len(exact_pairs)
    )
    print(
        "Additional rows excluded for exact-pixel overlap:",
        len(duplicate_ids)
    )
    print(
        "Ambiguous pHash pairs requiring caution:",
        len(unresolved_pairs)
    )
    print("Prepared splits:", args.out.resolve())
    print("Summary:", report_path.resolve())
    print("Original ZIPs and source images were not changed.")


if __name__ == "__main__":
    main()
