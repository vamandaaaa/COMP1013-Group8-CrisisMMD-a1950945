"""Extract frozen BERT token features for CrisisMMD.

Uses cleaned train/dev only. Original test is not accessed.
The token features will be used for cross-attention.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def main():
    parser = argparse.ArgumentParser(
        description="Extract frozen BERT token features"
    )

    parser.add_argument(
        "--processed-dir", type=Path,
        default=Path("data/processed")
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("data/features")
    )
    parser.add_argument(
        "--model", default="bert-base-uncased"
    )
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--device", choices=["cpu", "mps"], default="cpu"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if not 4 <= args.max_length <= 512:
        parser.error("Invalid maximum token length")

    if args.batch_size < 1 or args.limit < 0:
        parser.error("Invalid batch size or limit")

    if args.limit and args.out_dir == Path("data/features"):
        if not args.check_only:
            parser.error(
                "Use --out-dir data/features/smoke "
                "when using --limit"
            )

    train = read_rows(
        args.processed_dir / "humanitarian_train.tsv"
    )
    dev = read_rows(
        args.processed_dir / "humanitarian_dev.tsv"
    )

    if not train or not dev:
        parser.error("Train/dev splits are missing")

    required = {"tweet_id", "tweet_text"}

    for rows in (train, dev):
        missing = required - set(rows[0])

        if missing:
            raise ValueError(
                f"Missing columns: {sorted(missing)}"
            )

    # One tweet may have multiple associated images.
    # Encode each unique tweet only once.
    text_by_id = {}

    for row in train + dev:
        tweet_id = row["tweet_id"]
        text = row["tweet_text"] or ""

        if tweet_id in text_by_id:
            if text_by_id[tweet_id] != text:
                raise ValueError(
                    "Conflicting text for tweet: " + tweet_id
                )

        text_by_id[tweet_id] = text

    ids = list(text_by_id)
    texts = list(text_by_id.values())

    if args.limit:
        ids = ids[:args.limit]
        texts = texts[:args.limit]

    print("Train:", len(train))
    print("Dev:", len(dev))
    print("Unique tweets:", len(ids))
    print("Original official test: NOT ACCESSED")

    if args.check_only:
        print("Input check passed. No model downloaded.")
        return

    if args.device == "mps":
        if not torch.backends.mps.is_available():
            parser.error("MPS unavailable. Use --device cpu")

    output = args.out_dir / "bert_token_features.npz"
    metadata = args.out_dir / "bert_token_metadata.json"

    if (output.exists() or metadata.exists()) and not args.overwrite:
        parser.error(
            "Output already exists. Use --overwrite "
            "only for intentional reruns."
        )

    from transformers import AutoModel, AutoTokenizer

    print("Loading BERT:", args.model)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)

    model = model.to(args.device).eval()

    hidden_size = int(model.config.hidden_size)

    if hidden_size != 768:
        raise ValueError("Expected 768 BERT dimensions")

    # Check how many tweets exceed our token limit.
    tokenized = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=False
    )

    token_lengths = [
        len(x) for x in tokenized["input_ids"]
    ]

    truncated = sum(
        length > args.max_length
        for length in token_lengths
    )

    print(
        f"Tweets exceeding {args.max_length} tokens:",
        truncated
    )

    features = np.empty(
        (len(ids), args.max_length, hidden_size),
        dtype=np.float16
    )

    masks = np.empty(
        (len(ids), args.max_length),
        dtype=np.uint8
    )

    with torch.inference_mode():
        for start in range(0, len(ids), args.batch_size):
            stop = min(
                start + args.batch_size, len(ids)
            )

            batch_texts = texts[start:stop]

            tokens = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt"
            )

            masks[start:stop] = (
                tokens["attention_mask"]
                .numpy()
                .astype(np.uint8)
            )

            inputs = {
                key: value.to(args.device)
                for key, value in tokens.items()
            }

            outputs = model(**inputs).last_hidden_state

            features[start:stop] = (
                outputs.float()
                .cpu()
                .numpy()
                .astype(np.float16)
            )

            if stop % 250 < args.batch_size or stop == len(ids):
                print(f"BERT: {stop}/{len(ids)} tweets")

    args.out_dir.mkdir(
        parents=True, exist_ok=True
    )

    temporary = output.with_suffix(".tmp.npz")

    np.savez_compressed(
        temporary,
        tweet_id=np.asarray(ids, dtype=str),
        feat=features,
        attention_mask=masks
    )

    temporary.replace(output)

    details = {
        "encoder": args.model,
        "model_revision": getattr(
            model.config, "_commit_hash", None
        ),
        "tokenizer": tokenizer.name_or_path,
        "max_length": args.max_length,
        "hidden_size": hidden_size,
        "unique_tweets": len(ids),
        "truncated_tweets": truncated,
        "feature_dtype": "float16",
        "frozen_encoder": True,
        "data": "Cleaned train/dev only",
        "official_test_accessed": False
    }

    metadata.write_text(
        json.dumps(details, indent=2) + "\n",
        encoding="utf-8"
    )

    print("Saved:", output)
    print("Metadata:", metadata)
    print("Feature shape:", features.shape)


if __name__ == "__main__":
    main()
