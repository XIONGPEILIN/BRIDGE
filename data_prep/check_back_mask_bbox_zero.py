#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union


DEFAULT_JSON = "/home/yanai-lab/xiong-p/ssd/xiong-p/qwenpe/dataset_qwen_pe_all.json"


Json = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


def _is_mapping(x: Any) -> bool:
    return isinstance(x, dict)


def _is_sequence(x: Any) -> bool:
    return isinstance(x, (list, tuple))


def _safe_str(x: Any, limit: int = 300) -> str:
    s = repr(x)
    if len(s) > limit:
        return s[: limit - 3] + "..."
    return s


def _record_id(rec: Any) -> str:
    if not _is_mapping(rec):
        return ""
    for k in ("id", "image_id", "img_id", "uid", "uuid", "name", "file_name", "image", "path"):
        v = rec.get(k)
        if isinstance(v, (str, int)):
            return f"{k}={v}"
    return ""


def iter_back_masks(obj: Any) -> Iterator[Tuple[str, Any]]:
    stack: List[Tuple[str, Any]] = [("$", obj)]
    while stack:
        path, cur = stack.pop()
        if _is_mapping(cur):
            for k, v in cur.items():
                p = f"{path}.{k}"
                if k == "back_mask":
                    yield (p, v)
                else:
                    stack.append((p, v))
        elif _is_sequence(cur):
            for i, v in enumerate(cur):
                stack.append((f"{path}[{i}]", v))


def load_records(path: str, force_jsonl: bool = False) -> Iterator[Any]:
    if force_jsonl:
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                s = line.strip()
                if not s:
                    continue
                try:
                    yield json.loads(s)
                except json.JSONDecodeError as e:
                    raise SystemExit(f"JSONL parse error at line {line_no}: {e}") from e
        return

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            # Fallback to JSONL
            f.seek(0)
            for line_no, line in enumerate(f, 1):
                s = line.strip()
                if not s:
                    continue
                try:
                    yield json.loads(s)
                except json.JSONDecodeError as e:
                    raise SystemExit(
                        "File is neither valid JSON nor JSONL. "
                        f"First failing line {line_no}: {e}"
                    ) from e
            return

    if isinstance(data, list):
        for rec in data:
            yield rec
    else:
        # single JSON object
        yield data


def resolve_path(p: str, root: str) -> str:
    if os.path.isabs(p):
        return p
    if root:
        return os.path.normpath(os.path.join(root, p))
    return os.path.normpath(p)


def compute_mask_bbox_xywh(mask_path: str) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int]]]:
    """
    Returns (bbox_xywh, (w,h)).
    - bbox_xywh: (x, y, w, h). If mask has no non-zero pixels, returns (0,0,0,0).
    - Returns (None, None) if the image cannot be read.
    """
    try:
        from PIL import Image  # type: ignore
    except Exception:
        raise SystemExit(
            "Missing dependency: Pillow. Install it or run in an environment with `PIL` available."
        )

    try:
        import numpy as np  # type: ignore
    except Exception:
        raise SystemExit("Missing dependency: numpy. Install it or run in an environment with numpy available.")

    try:
        with Image.open(mask_path) as img:
            img = img.convert("RGBA")
            w, h = img.size
            arr = np.array(img)
    except Exception:
        return (None, None)

    # Any non-zero in any channel counts as foreground.
    fg = (arr[:, :, 0] != 0) | (arr[:, :, 1] != 0) | (arr[:, :, 2] != 0) | (arr[:, :, 3] != 0)
    if not fg.any():
        return ((0, 0, 0, 0), (w, h))

    ys, xs = np.where(fg)
    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    bbox = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
    return (bbox, (w, h))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Read dataset JSON/JSONL, resolve each item's `back_mask` mask-image path, "
            "compute bbox from non-zero pixels, and report whether bbox becomes (0,0,0,0)."
        )
    )
    ap.add_argument("--json", dest="json_path", default=DEFAULT_JSON, help=f"dataset json path (default: {DEFAULT_JSON})")
    ap.add_argument("--jsonl", action="store_true", help="treat input as JSONL")
    ap.add_argument(
        "--root",
        default=None,
        help="prefix for relative mask paths (default: dirname of --json)",
    )
    ap.add_argument("--max-samples", type=int, default=20, help="max examples to print")
    ap.add_argument("--check-missing", action="store_true", help="count missing/unreadable mask files")
    args = ap.parse_args()

    json_path = args.json_path
    if not os.path.exists(json_path):
        print(f"ERROR: json not found: {json_path}", file=sys.stderr)
        return 2

    root = args.root
    if root is None:
        root = os.path.dirname(os.path.abspath(json_path))

    total_records = 0
    records_with_back_mask = 0
    back_mask_paths = 0
    mask_missing_or_unreadable = 0

    bbox_all_zero = 0
    bbox_non_zero = 0

    examples_zero: List[str] = []
    examples_nonzero: List[str] = []
    examples_missing: List[str] = []

    for rec in load_records(json_path, force_jsonl=args.jsonl):
        total_records += 1
        rec_tag = _record_id(rec)

        saw_back_mask = False
        for bm_path, bm in iter_back_masks(rec):
            # In your dataset, back_mask is typically a string path to a PNG mask.
            if isinstance(bm, str):
                saw_back_mask = True
                back_mask_paths += 1
                mask_path = resolve_path(bm, root)
                if args.check_missing and not os.path.exists(mask_path):
                    mask_missing_or_unreadable += 1
                    if len(examples_missing) < args.max_samples:
                        examples_missing.append(f"rec#{total_records} {rec_tag} {bm_path}: missing {mask_path}")
                    continue

                bbox, wh = compute_mask_bbox_xywh(mask_path)
                if bbox is None or wh is None:
                    if args.check_missing:
                        mask_missing_or_unreadable += 1
                        if len(examples_missing) < args.max_samples:
                            examples_missing.append(
                                f"rec#{total_records} {rec_tag} {bm_path}: unreadable {mask_path}"
                            )
                    continue

                if bbox == (0, 0, 0, 0):
                    bbox_all_zero += 1
                    if len(examples_zero) < args.max_samples:
                        examples_zero.append(
                            f"rec#{total_records} {rec_tag} {bm_path}: bbox={bbox} size={wh} path={mask_path}"
                        )
                else:
                    bbox_non_zero += 1
                    if len(examples_nonzero) < args.max_samples:
                        examples_nonzero.append(
                            f"rec#{total_records} {rec_tag} {bm_path}: bbox={bbox} size={wh} path={mask_path}"
                        )
            else:
                # If back_mask isn't a string, we don't assume its schema here.
                # Print a short sample to help you adjust parsing if needed.
                saw_back_mask = True
                if len(examples_missing) < args.max_samples:
                    examples_missing.append(
                        f"rec#{total_records} {rec_tag} {bm_path}: unsupported back_mask type {type(bm).__name__} value={_safe_str(bm)}"
                    )

        if saw_back_mask:
            records_with_back_mask += 1

    print("== Summary ==")
    print(f"json: {json_path}")
    print(f"root: {root}")
    print(f"records: {total_records}")
    print(f"records_with_back_mask: {records_with_back_mask}")
    print(f"back_mask_string_paths: {back_mask_paths}")
    if args.check_missing:
        print(f"mask_missing_or_unreadable: {mask_missing_or_unreadable}")
    print(f"computed_bbox_all_zero(0,0,0,0): {bbox_all_zero}")
    print(f"computed_bbox_non_zero: {bbox_non_zero}")

    if examples_zero:
        print("\n== Examples: bbox all zero ==")
        print("\n".join(examples_zero))
    if examples_missing:
        print("\n== Examples: missing/unreadable/unsupported ==")
        print("\n".join(examples_missing))
    if examples_nonzero:
        print("\n== Examples: bbox non-zero ==")
        print("\n".join(examples_nonzero))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
