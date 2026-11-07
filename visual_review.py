"""Create contact sheets for reviewing CrisisMMD images."""

import argparse
import csv
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageOps


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def image_paths(splits_zip):
    """Get image paths from the dataset metadata."""
    paths = {}

    with ZipFile(splits_zip) as archive:
        for name in archive.namelist():
            if not name.endswith(".tsv"):
                continue

            if name.startswith("__MACOSX/"):
                continue

            with archive.open(name) as source:
                text = io.TextIOWrapper(
                    source, encoding="utf-8-sig"
                )
                rows = csv.DictReader(text, delimiter="\t")

                if not rows.fieldnames:
                    continue

                if not {"image_id", "image"}.issubset(rows.fieldnames):
                    continue

                for row in rows:
                    if row["image_id"] and row["image"]:
                        paths.setdefault(
                            row["image_id"], row["image"]
                        )

    return paths


def find_images(root):
    """Index local images using their image IDs."""
    images = {}

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in (
            ".jpg", ".jpeg", ".png", ".webp"
        ):
            images.setdefault(path.stem, path)

    return images


def get_image_path(image_id, root, paths, index):
    relative = Path(paths.get(image_id, ""))

    if (
        str(relative) != "."
        and not relative.is_absolute()
        and ".." not in relative.parts
    ):
        for candidate in (
            root / relative,
            root.parent / relative
        ):
            if candidate.is_file():
                return candidate

    return index.get(image_id)


def paste_thumbnail(canvas, path, x, y, width=230, height=150):
    """Add a resized image to a contact sheet."""
    draw = ImageDraw.Draw(canvas)

    draw.rectangle(
        (x, y, x + width, y + height),
        outline="gray"
    )

    if path is None:
        draw.text(
            (x + 10, y + 55),
            "IMAGE NOT FOUND",
            fill="red"
        )
        return False

    try:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((width - 4, height - 4))

            canvas.paste(
                image,
                (
                    x + (width - image.width) // 2,
                    y + (height - image.height) // 2
                )
            )

        return True

    except (OSError, ValueError):
        draw.text(
            (x + 10, y + 55),
            "UNREADABLE IMAGE",
            fill="red"
        )
        return False


def select_pairs(pairs, limit):
    """Prioritise label conflicts and cover different distances."""
    def pair_order(pair):
        return (
            int(pair["phash_distance"]),
            pair["image_id_A"]
        )

    conflicts = sorted(
        [p for p in pairs if p["label_conflict"] == "1"],
        key=pair_order
    )

    chosen = conflicts[:limit]
    selected_ids = {id(p) for p in chosen}

    buckets = defaultdict(list)

    for pair in sorted(pairs, key=pair_order):
        if id(pair) not in selected_ids:
            distance = int(pair["phash_distance"])
            buckets[distance].append(pair)

    while len(chosen) < limit and any(buckets.values()):
        for distance in sorted(buckets):
            if len(chosen) < limit and buckets[distance]:
                chosen.append(buckets[distance].pop(0))

    return chosen


def group_sheet(group_number, rows, resolve, output):
    """Show nine examples from one repeated-feature group."""
    def group_order(row):
        return (
            row["split"] == "outside_agreed_humanitarian",
            {"train": 0, "dev": 1, "test": 2}.get(
                row["split"], 3
            ),
            row["image_id"]
        )

    shown = sorted(rows, key=group_order)[:9]

    canvas = Image.new(
        "RGB",
        (780, 62 + math.ceil(len(shown) / 3) * 215),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    draw.text(
        (12, 12),
        f"Feature group {group_number} | {len(rows)} matching vectors",
        fill="black"
    )

    draw.text(
        (12, 33),
        "Inspect images before deciding whether they are placeholders.",
        fill="gray"
    )

    missing = set()

    for i, row in enumerate(shown):
        x = 14 + (i % 3) * 260
        y = 68 + (i // 3) * 215

        path = resolve(row["image_id"])

        if not paste_thumbnail(canvas, path, x, y):
            missing.add(row["image_id"])

        draw.text(
            (x, y + 155),
            row["image_id"],
            fill="black"
        )

        draw.text(
            (x, y + 174),
            row["split"],
            fill="gray"
        )

    name = f"feature_group_{group_number}.jpg"
    canvas.save(output / name, quality=90)

    return name, missing


def pair_sheets(pairs, resolve, output):
    """Show image pairs for visual comparison."""
    files = []
    missing = set()

    for start in range(0, len(pairs), 12):
        page = pairs[start:start + 12]

        canvas = Image.new(
            "RGB",
            (1100, 42 + math.ceil(len(page) / 2) * 242),
            "white"
        )

        draw = ImageDraw.Draw(canvas)

        draw.text(
            (12, 12),
            f"pHash pairs {start + 1} to {start + len(page)}",
            fill="black"
        )

        for i, pair in enumerate(page):
            x = 10 + (i % 2) * 550
            y = 49 + (i // 2) * 242

            conflict = pair["label_conflict"] == "1"
            status = "LABEL CONFLICT" if conflict else "same label"

            draw.text(
                (x, y),
                f"Pair {start+i+1} | pHash={pair['phash_distance']} | {status}",
                fill="red" if conflict else "black"
            )

            for side, offset in (("A", 0), ("B", 270)):
                image_id = pair[f"image_id_{side}"]
                px = x + offset

                if not paste_thumbnail(
                    canvas,
                    resolve(image_id),
                    px,
                    y + 22,
                    250,
                    150
                ):
                    missing.add(image_id)

                draw.text(
                    (px, y + 177),
                    f"{side} {pair[f'split_{side}']}: {image_id}",
                    fill="black"
                )

                draw.text(
                    (px, y + 195),
                    pair[f"label_{side}"][:37],
                    fill="gray"
                )

        name = f"phash_pairs_{start // 12 + 1:02d}.jpg"
        canvas.save(output / name, quality=88)
        files.append(name)

    return files, missing


def main():
    parser = argparse.ArgumentParser(
        description="CrisisMMD local visual review"
    )

    parser.add_argument(
        "--splits-zip", type=Path, default=Path("splits.zip")
    )
    parser.add_argument(
        "--audit-dir", type=Path, default=Path("audit_output")
    )
    parser.add_argument(
        "--images-root", type=Path, default=Path("data/raw")
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("audit_output/visual_review")
    )
    parser.add_argument(
        "--max-pairs", type=int, default=60
    )

    args = parser.parse_args()

    if not args.images_root.is_dir():
        parser.error("Image directory not found.")

    if args.max_pairs < 1:
        parser.error("--max-pairs must be positive")

    args.out.mkdir(parents=True, exist_ok=True)

    paths = image_paths(args.splits_zip)
    index = find_images(args.images_root)

    def resolve(image_id):
        return get_image_path(
            image_id, args.images_root, paths, index
        )

    feature_rows = read_csv(
        args.audit_dir / "identical_image_feature_groups.csv"
    )

    all_pairs = read_csv(
        args.audit_dir / "phash_candidates_for_review.csv"
    )

    groups = defaultdict(list)

    for row in feature_rows:
        number = int(row["feature_group"])

        if number <= 3:
            groups[number].append(row)

    group_files = []
    missing = set()

    for number in sorted(groups):
        name, not_found = group_sheet(
            number, groups[number], resolve, args.out
        )

        group_files.append(name)
        missing.update(not_found)

    chosen = select_pairs(
        all_pairs,
        min(args.max_pairs, len(all_pairs))
    )

    pair_files, not_found = pair_sheets(
        chosen, resolve, args.out
    )

    missing.update(not_found)

    review_rows = []

    for number, row in enumerate(chosen, 1):
        review_rows.append({
            "review_number": number,
            **row,
            "image_path_A": paths.get(row["image_id_A"], ""),
            "image_path_B": paths.get(row["image_id_B"], ""),
            "review_status": "",
            "notes": ""
        })

    if review_rows:
        with open(
            args.out / "priority_pair_review.csv",
            "w",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(review_rows[0])
            )
            writer.writeheader()
            writer.writerows(review_rows)

    report = {
        "status": "visual_review_only_no_splits_changed",
        "local_images_indexed": len(index),
        "metadata_image_paths": len(paths),
        "feature_groups_reviewed": [
            {
                "group": number,
                "total_cached_images": len(groups[number]),
                "agreed_task_rows": sum(
                    row["split"] != "outside_agreed_humanitarian"
                    for row in groups[number]
                )
            }
            for number in sorted(groups)
        ],
        "pairs_available": len(all_pairs),
        "pairs_in_review": len(chosen),
        "pair_sheets": pair_files,
        "group_sheets": group_files,
        "missing_images_from_review": len(missing),
        "missing_review_ids_first_10": sorted(missing)[:10],
        "note": (
            "Visual review only. No images were labelled or deleted."
        )
    }

    (args.out / "review_summary.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8"
    )

    print("CrisisMMD — Visual Review")
    print("Indexed images:", len(index))
    print(
        "Feature groups:",
        [len(groups[n]) for n in sorted(groups)]
    )
    print(
        f"pHash pairs selected: {len(chosen)} of {len(all_pairs)}"
    )
    print(
        f"Contact sheets: {len(group_files)} groups "
        f"+ {len(pair_files)} pair sheets"
    )
    print("Missing images:", len(missing))
    print("Saved in:", args.out.resolve())
    print("No original data was changed.")


if __name__ == "__main__":
    main()
