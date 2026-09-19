#!/usr/bin/env python3
"""Precompute a deterministic sort+greedy batched training order and save it to JSON.

Two operating modes:

1. **Cache mode** (recommended, set `--cache_dir`): reads `(crop_h, crop_w)` from each
   shard's `manifest.json`'s `subcrop.pixels_size` field. Fast (~5 s for 30k shards),
   no mask PNG reads needed.

2. **Raw mode** (default, no `--cache_dir`): falls back to instantiating
   `SubjectDrivenFlux2Dataset`, which reads every mask PNG to compute bucket keys.

Algorithm: **sort + greedy packing** (Method A from the design doc).
   - sort all non-empty samples by `crop_h * crop_w` ascending
   - chunk into consecutive groups of `train_batch_size` -> each batch contains
     similar-sized samples (minimises sub-branch padding overhead)
   - optionally shuffle in larger blocks (`--shuffle_block`) to break sampling
     correlation while keeping padding overhead small
   - **no drop**: the final under-sized batch is kept; the training-time collate
     pads it to `train_batch_size` via the standard sub-pad path, but the batch
     itself is smaller than usual (acceptable; effective batch will be smaller
     for one step per epoch)

Output schema:

    {
      "meta": { ... },
      "batches": [[i0, i1, ...], ...]   # list[list[int]] dataset-index batches
    }

Training loads via `--bucket_order_file` and replays in order every epoch.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from accelerate import PartialState  # noqa: E402

PartialState()

from train_flux2_klein_subject_img2img_sub_ids20 import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_SOURCE_BASE,
    DEFAULT_TARGET_BASE,
    CachedSubjectDrivenFlux2Dataset,
    SubjectDrivenFlux2Dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cache_dir", type=Path, default=None,
                        help="If set, read bucket keys from cache manifests instead of masks.")
    parser.add_argument("--target_base", type=Path, default=DEFAULT_TARGET_BASE)
    parser.add_argument("--source_base", type=Path, default=DEFAULT_SOURCE_BASE)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--downsample_factor", type=int, default=16)
    parser.add_argument("--train_batch_size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sub_region_mode", choices=("bbox", "mask"), default="mask",
                        help="Only used in raw mode.")
    parser.add_argument(
        "--shuffle_block",
        type=int,
        default=64,
        help="After sort+chunk, shuffle batches in contiguous blocks of this size. "
             "Higher = more sampling diversity but more padding overhead. "
             "Set 1 to disable shuffling (pure strict sort order); set 0 for full shuffle.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <dataset_stem>_bucket_order_res{R}_ds{D}_bs{B}_seed{S}.json.",
    )
    return parser.parse_args()


def sort_and_pack(
    keys_by_index: list[tuple[int, int]],
    batch_size: int,
    shuffle_block: int,
    rng: random.Random,
) -> tuple[list[list[int]], list[int]]:
    """Sort all non-empty samples by sub_seq_len, pack into batches, then block-shuffle.

    Returns:
        batches: list of dataset-index lists
        max_n_sub_per_batch: same length as batches
    """
    items = [(i, k[0] * k[1], k) for i, k in enumerate(keys_by_index) if k != (0, 0)]
    # Stable sort: by area, then by h, then by w, for reproducibility
    items.sort(key=lambda t: (t[1], t[2][0], t[2][1]))

    batches: list[list[int]] = []
    max_n_sub: list[int] = []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        if not chunk:
            continue
        batches.append([t[0] for t in chunk])
        max_n_sub.append(max(t[1] for t in chunk))

    # Block shuffle: permute within contiguous blocks of `shuffle_block` batches.
    if shuffle_block == 0:
        # Full global shuffle
        order = list(range(len(batches)))
        rng.shuffle(order)
        batches = [batches[i] for i in order]
        max_n_sub = [max_n_sub[i] for i in order]
    elif shuffle_block > 1:
        for start in range(0, len(batches), shuffle_block):
            end = min(start + shuffle_block, len(batches))
            order = list(range(start, end))
            rng.shuffle(order)
            batches[start:end] = [batches[i] for i in order]
            max_n_sub[start:end] = [max_n_sub[i] for i in order]
    # shuffle_block == 1: keep strict sorted order

    return batches, max_n_sub


def _block_shuffle(
    batches: list[list[int]],
    max_n_sub: list[int],
    shuffle_block: int,
    rng: random.Random,
) -> tuple[list[list[int]], list[int]]:
    """Shuffle at BATCH granularity (never mixes samples across batches)."""
    if shuffle_block == 0:
        order = list(range(len(batches)))
        rng.shuffle(order)
        return [batches[i] for i in order], [max_n_sub[i] for i in order]
    if shuffle_block > 1:
        for start in range(0, len(batches), shuffle_block):
            end = min(start + shuffle_block, len(batches))
            order = list(range(start, end))
            rng.shuffle(order)
            batches[start:end] = [batches[i] for i in order]
            max_n_sub[start:end] = [max_n_sub[i] for i in order]
    return batches, max_n_sub


def group_and_pack(
    main_keys: list[tuple[int, int]],
    sub_keys: list[tuple[int, int]],
    batch_size: int,
    shuffle_block: int,
    rng: random.Random,
) -> tuple[list[list[int]], list[int]]:
    """Group samples by MAIN resolution, then within each group sort by sub_seq_len and pack.

    REQUIRED for train_batch_size > 1 on native-aspect data: collate_cached_examples
    hard-`torch.stack`s packed_target / packed_main / cond_image_ids, so every sample in a
    batch MUST share the same main token count. The sub branch is padded to batch max, so sub
    crop size may differ within a batch (sorting by sub area just minimises that padding).

    Per-group remainder batches (size < batch_size) are KEPT (no drop) — they are still
    main-homogeneous, so the collate stacks them fine (effective batch is just smaller for one
    step per group). Batch-granularity block shuffle preserves main-homogeneity.
    """
    from collections import defaultdict

    groups: dict[tuple[int, int], list[tuple[int, int, tuple[int, int]]]] = defaultdict(list)
    for i, (mk, sk) in enumerate(zip(main_keys, sub_keys)):
        if sk == (0, 0) or mk == (0, 0):  # empty mask or no cache shard -> skip
            continue
        groups[mk].append((i, sk[0] * sk[1], sk))

    batches: list[list[int]] = []
    max_n_sub: list[int] = []
    for mk in sorted(groups.keys()):  # deterministic group order
        items = groups[mk]
        items.sort(key=lambda t: (t[1], t[2][0], t[2][1]))
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            batches.append([t[0] for t in chunk])
            max_n_sub.append(max(t[1] for t in chunk))

    return _block_shuffle(batches, max_n_sub, shuffle_block, rng)


def main() -> None:
    args = parse_args()

    if args.cache_dir is not None:
        dataset = CachedSubjectDrivenFlux2Dataset(
            dataset_path=args.dataset,
            cache_dir=args.cache_dir,
            precompute_bucket_keys=True,
        )
        mode = "cache"
    else:
        dataset = SubjectDrivenFlux2Dataset(
            dataset_path=args.dataset,
            target_base=args.target_base,
            source_base=args.source_base,
            resolution=args.resolution,
            sub_region_mode=args.sub_region_mode,
            downsample_factor=args.downsample_factor,
            precompute_bucket_keys=True,
        )
        mode = "raw"

    keys = [tuple(k) for k in dataset.bucket_keys]
    main_keys_raw = getattr(dataset, "main_keys", None)
    main_keys = [tuple(k) for k in main_keys_raw] if main_keys_raw is not None else None
    skipped_empty = sum(1 for k in keys if k == (0, 0))

    rng = random.Random(args.seed)
    if main_keys is not None and len(main_keys) == len(keys):
        # Group by MAIN resolution so every batch is stackable (required for bs>1, native aspect).
        batches, max_n_sub_per_batch = group_and_pack(
            main_keys, keys, args.train_batch_size, args.shuffle_block, rng
        )
        grouping = "main_resolution"
        # Hard sanity check: every batch must be homogeneous in main resolution.
        for bi, b in enumerate(batches):
            mks = {main_keys[i] for i in b}
            if len(mks) != 1:
                raise AssertionError(
                    f"Batch {bi} spans multiple main resolutions {mks} — collate torch.stack "
                    f"would crash. group_and_pack is broken."
                )
        n_main_buckets = len({main_keys[i] for b in batches for i in b})
    else:
        if args.train_batch_size > 1:
            print(
                "WARNING: no per-sample main_keys (raw mode) — batches are NOT grouped by main "
                "resolution. train_batch_size>1 on native-aspect data WILL fail to torch.stack "
                "packed_target. Use --cache_dir (cache mode) for bs>1."
            )
        batches, max_n_sub_per_batch = sort_and_pack(keys, args.train_batch_size, args.shuffle_block, rng)
        grouping = "sub_only"
        n_main_buckets = 0

    n_used = sum(len(b) for b in batches)
    avg_max = sum(max_n_sub_per_batch) / max(len(max_n_sub_per_batch), 1)
    # Padding overhead = (sum of max * batch_size - sum of actual sub_seq) / (sum of max * batch_size)
    total_padded = sum(m * len(b) for m, b in zip(max_n_sub_per_batch, batches))
    total_actual = sum(keys[i][0] * keys[i][1] for b in batches for i in b)
    pad_overhead = 1.0 - total_actual / max(total_padded, 1)

    output = args.output
    if output is None:
        suffix = f"_bucket_order_res{args.resolution}_ds{args.downsample_factor}_bs{args.train_batch_size}_seed{args.seed}.json"
        output = args.dataset.with_name(args.dataset.stem + suffix)

    payload = {
        "meta": {
            "dataset": str(args.dataset.resolve()),
            "mode": mode,
            "resolution": args.resolution,
            "downsample_factor": args.downsample_factor,
            "train_batch_size": args.train_batch_size,
            "seed": args.seed,
            "shuffle_block": args.shuffle_block,
            "algorithm": "group_by_main_resolution+sort+greedy" if grouping == "main_resolution" else "sort+greedy",
            "grouping": grouping,
            "n_main_buckets": n_main_buckets,
            "n_samples_total": len(dataset),
            "n_samples_empty_mask_skipped": skipped_empty,
            "n_samples_used": n_used,
            "n_batches": len(batches),
            "avg_max_sub_seq_len": round(avg_max, 1),
            "padding_overhead_fraction": round(pad_overhead, 4),
            "max_sub_seq_len_per_batch": max_n_sub_per_batch,
        },
        "batches": batches,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(payload, f)

    print(f"Mode               : {mode}")
    print(f"Grouping           : {grouping}" + (f" ({n_main_buckets} main-resolution buckets)" if grouping == "main_resolution" else ""))
    print(f"Dataset            : {args.dataset}")
    print(f"Samples total      : {len(dataset)}")
    print(f"Samples skipped    : {skipped_empty} (empty mask)")
    print(f"Samples used       : {n_used} (no drop in sort+greedy)")
    print(f"Batches (bs={args.train_batch_size})   : {len(batches)}")
    print(f"Avg batch max N_sub: {avg_max:.1f}")
    print(f"Padding overhead   : {pad_overhead*100:.2f}%")
    print(f"Shuffle block      : {args.shuffle_block}")
    print(f"Written to         : {output}")


if __name__ == "__main__":
    main()
