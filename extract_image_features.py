"""Extract frozen ResNet-50 or ViT-B/16 image features for CrisisMMD.

Uses cleaned train and dev rows only. Original test is not accessed.
ViT also saves 16 patch vectors per image for later fusion.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn
from torchvision.models import (
    ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16
)


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def image_path(images_root, name):
    root = images_root.resolve()
    path = (root / name).resolve()

    if not path.is_relative_to(root):
        raise ValueError("Unsafe image path: " + str(name))

    return path


def load_images(rows, images_root, transform):
    for row in rows:
        path = image_path(images_root, row["image"])

        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            tensor = transform(image)

        yield row["image_id"], tensor


def build_model(name):
    if name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        model.fc = nn.Identity()
    else:
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
        model.heads = nn.Identity()

    return model, weights.transforms(), weights.name


def encode_batch(model, batch, name):
    if name == "resnet50":
        pooled = model(batch)
        return pooled, None

    # Extract ViT CLS and patch features.
    tokens = model._process_input(batch)

    cls = model.class_token.expand(batch.shape[0], -1, -1)
    encoded = model.encoder(torch.cat([cls, tokens], dim=1))

    pooled = encoded[:, 0]

    # Reduce the 14x14 patch grid to 4x4.
    patches = encoded[:, 1:].transpose(1, 2).reshape(
        batch.shape[0], 768, 14, 14
    )

    patches = F.adaptive_avg_pool2d(patches.cpu(), (4, 4))
    patches = patches.flatten(2).transpose(1, 2)

    return pooled, patches


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["resnet50", "vit_b_16"],
        required=True
    )
    parser.add_argument(
        "--processed-dir", type=Path,
        default=Path("data/processed")
    )
    parser.add_argument(
        "--images-root", type=Path,
        default=Path("data/raw")
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("data/features")
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps"],
        default="cpu"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Smoke test only"
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.batch_size < 1 or args.limit < 0:
        parser.error("Invalid batch size or limit")

    if args.limit and args.out_dir == Path("data/features"):
        parser.error(
            "Use --out-dir data/features/smoke when using --limit"
        )

    train = read_rows(
        args.processed_dir / "humanitarian_train.tsv"
    )
    dev = read_rows(
        args.processed_dir / "humanitarian_dev.tsv"
    )

    if not train or not dev:
        parser.error("Cleaned train/dev splits are missing or empty")

    # Keep one feature vector per unique image ID.
    rows = []
    seen = set()

    for row in train + dev:
        if row["image_id"] not in seen:
            rows.append(row)
            seen.add(row["image_id"])

    if args.limit:
        rows = rows[:args.limit]

    missing = [
        r["image"] for r in rows
        if not image_path(
            args.images_root, r["image"]
        ).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} images missing. First: {missing[0]}"
        )

    print("Train:", len(train))
    print("Dev:", len(dev))
    print("Images to extract:", len(rows))
    print("Original official test: NOT ACCESSED")

    if args.check_only:
        print("Image paths verified. No weights downloaded.")
        return

    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS unavailable. Use --device cpu")

    device = torch.device(args.device)

    output = args.out_dir / (args.model + "_features.npz")
    metadata = args.out_dir / (args.model + "_metadata.json")

    if output.exists() and not args.overwrite:
        parser.error(
            "Output exists. Use --overwrite for intentional reruns."
        )

    model, transform, weight_name = build_model(args.model)
    model = model.to(device).eval()

    ids = []
    features = []
    patch_vectors = []

    with torch.inference_mode():
        iterator = iter(
            load_images(rows, args.images_root, transform)
        )
        done = 0

        while done < len(rows):
            image_ids = []
            tensors = []

            count = min(args.batch_size, len(rows) - done)

            for _ in range(count):
                image_id, tensor = next(iterator)
                image_ids.append(image_id)
                tensors.append(tensor)

            batch = torch.stack(tensors).to(device)

            pooled, patches = encode_batch(
                model, batch, args.model
            )

            ids.extend(image_ids)

            features.append(
                pooled.cpu().numpy().astype(np.float32)
            )

            if patches is not None:
                patch_vectors.append(
                    patches.cpu().numpy().astype(np.float16)
                )

            done += len(image_ids)

            if done % 100 < args.batch_size or done == len(rows):
                print(
                    f"{args.model}: {done}/{len(rows)} images"
                )

    array = np.concatenate(features, axis=0)

    payload = {
        "image_id": np.array(ids, dtype=str),
        "feat": array
    }

    if patch_vectors:
        payload["patch_feat"] = np.concatenate(
            patch_vectors, axis=0
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(output)

    details = {
        "model": args.model,
        "weights": weight_name,
        "torchvision_version": __import__(
            "torchvision"
        ).__version__,
        "splits": "cleaned train + dev only",
        "num_images": len(ids),
        "feature_shape": list(array.shape),
        "patch_shape": (
            list(payload["patch_feat"].shape)
            if "patch_feat" in payload else None
        ),
        "preprocessing": "TorchVision pretrained weight transforms",
        "frozen_encoder": True
    }

    metadata.write_text(
        json.dumps(details, indent=2) + "\n",
        encoding="utf-8"
    )

    print("Saved:", output)
    print("Metadata:", metadata)


if __name__ == "__main__":
    main()
