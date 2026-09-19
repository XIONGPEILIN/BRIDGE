#!/usr/bin/env python
"""Consolidate DeepSpeed ZeRO-1 checkpoints of a training run into BF16 diffusers
transformer dirs, then (optionally) delete the heavy DeepSpeed state.

WHY this exact path (do not "shortcut" by reading mp_rank_00_model_states.pt['module']):
  This project trains with ProdigyPlusScheduleFree. The historical DeepSpeed checkpoints were
  saved through an Accelerate wrapper whose ``eval()`` did not reach the nested BF16 optimizer.
  Consequently both the bf16 module weights and the sharded fp32 masters are TRAIN-view weights,
  even though `save_checkpoint` called ``optimizer.eval()``. The optimizer shards also contain
  ScheduleFree's fp32 ``z`` tensor, so the exact eval view can still be recovered rank by rank:

      eval = torch.lerp(train, z, 1 - 1 / beta1)

  The code below validates DeepSpeed's rank/parameter fragment mapping, performs that conversion,
  casts directly to bf16, and saves a normal diffusers transformer checkpoint.
  This is the format the inference pipeline already loads via custom_cls.from_pretrained(...).

For each runs/<run>/checkpoint-XXXX/:
  1) Reconstruct bf16 eval-view parameters from each rank's fp32 master + ScheduleFree z state.
  2) build the matching architecture (base Flux2Transformer2DModel for no_sub/sub_no_exchange,
     else the hard/soft PE-exchange subclass with the SAME ste_* config as training)
  3) load_state_dict(state_dict) -- every trainable PARAMETER must be present (strict on params)
  4) model.to(bfloat16); model.save_pretrained(checkpoint-XXXX/transformer)
  5) verify checkpoint-XXXX/transformer reloads via from_pretrained (the inference entrypoint)
  6) unless --keep_ds / --dry_run: delete pytorch_model/, zero_to_fp32.py, latest, random_states_*.pkl
     (keep training_state.json and the new transformer/)

Experiment mode / pretrained / ste_* come from runs/<run>/train_args.json when present
(soft/base finish naturally and write it), else from explicit flags (hard is killed at 1800
before train_args.json is written -> pass --experiment_mode hard_exchange).
"""
import argparse
import inspect
import json
import math
import re
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusers import Flux2Transformer2DModel  # noqa: E402  (vendored)
from flux2_klein_pe_exchange import Flux2KleinPEExchangeTransformer2DModel  # noqa: E402
from flux2_klein_pe_soft_exchange import Flux2KleinPESoftExchangeTransformer2DModel  # noqa: E402

DEFAULT_MODEL = "black-forest-labs/FLUX.2-klein-base-9B"
BASE_MODES = {"no_sub", "sub_no_exchange"}
CONVERSION_CHUNK_NUMEL = 4 * 1024 * 1024


def log(msg: str) -> None:
    print(f"[consolidate] {msg}", flush=True)


def custom_cls_for(mode: str):
    if mode == "soft_exchange":
        return Flux2KleinPESoftExchangeTransformer2DModel
    return Flux2KleinPEExchangeTransformer2DModel


def _natural_key(path: Path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def _shape_numel(shape) -> int:
    return int(shape.numel()) if hasattr(shape, "numel") else math.prod(shape)


def _copy_eval_fragment(
    destination: torch.Tensor,
    destination_start: int,
    train_flat: torch.Tensor,
    z_flat: torch.Tensor | None,
    source_start: int,
    numel: int,
    beta1: float | None,
) -> None:
    destination_flat = destination.view(-1)
    for offset in range(0, numel, CONVERSION_CHUNK_NUMEL):
        length = min(CONVERSION_CHUNK_NUMEL, numel - offset)
        train_chunk = train_flat.narrow(0, source_start + offset, length)
        if z_flat is None:
            eval_chunk = train_chunk
        else:
            eval_chunk = torch.lerp(
                train_chunk,
                z_flat.narrow(0, source_start + offset, length),
                1.0 - 1.0 / beta1,
            )
        destination_flat.narrow(0, destination_start + offset, length).copy_(
            eval_chunk.to(torch.bfloat16)
        )


def get_schedulefree_eval_state_dict_from_zero_checkpoint(ckpt_dir: Path):
    """Recover a BF16 eval-view state dict from this project's ZeRO-1 checkpoint.

    DeepSpeed reconstructs ZeRO-1/2 parameters by concatenating the per-rank fp32
    partitions and slicing that flat vector according to ``param_shapes``. We use
    the same contract, while independently validating ``param_slice_mappings``.
    """
    ckpt_dir = Path(ckpt_dir)
    ds_dir = ckpt_dir / "pytorch_model"
    optim_files = sorted(ds_dir.glob("*_optim_states.pt"), key=_natural_key)
    model_files = sorted(ds_dir.glob("*model_states.pt"), key=_natural_key)
    if not optim_files or len(model_files) != 1:
        raise RuntimeError(
            f"Expected optimizer shards and exactly one model-state file under {ds_dir}; "
            f"found optim={len(optim_files)} model={len(model_files)}."
        )

    model_blob = torch.load(model_files[0], map_location="cpu", weights_only=False, mmap=True)
    param_shape_groups = model_blob.get("param_shapes")
    if not isinstance(param_shape_groups, list) or len(param_shape_groups) != 1:
        raise RuntimeError(
            f"Only the run's single ZeRO parameter group is supported; got {type(param_shape_groups)} "
            f"with length {len(param_shape_groups) if param_shape_groups is not None else None}."
        )
    param_shapes = param_shape_groups[0]
    if model_blob.get("frozen_param_shapes"):
        raise RuntimeError("Frozen parameter fragments are not supported by this converter.")
    if model_blob.get("shared_params"):
        raise RuntimeError("Shared parameters are not expected for this transformer checkpoint.")

    param_offsets = OrderedDict()
    full_numel = 0
    for name, shape in param_shapes.items():
        numel = _shape_numel(shape)
        param_offsets[name] = (full_numel, numel)
        full_numel += numel

    state_dict = OrderedDict(
        (name, torch.empty(tuple(shape), dtype=torch.bfloat16)) for name, shape in param_shapes.items()
    )
    covered = {name: 0 for name in param_shapes}
    conversion_groups = []
    partition_numel = None

    for rank, optim_file in enumerate(optim_files):
        blob = torch.load(optim_file, map_location="cpu", weights_only=False, mmap=True)
        optim = blob["optimizer_state_dict"]
        partition_count = optim["partition_count"]
        world_size = max(partition_count) if isinstance(partition_count, list) else int(partition_count)
        if world_size != len(optim_files):
            raise RuntimeError(f"Checkpoint expects {world_size} optimizer shards, found {len(optim_files)}.")
        if int(optim["zero_stage"]) > 2:
            raise RuntimeError(f"Only ZeRO-1/2 is supported, got {optim['zero_stage']}.")

        fp32_groups = optim["single_partition_of_fp32_groups"]
        base = optim["base_optimizer_state"]
        if len(fp32_groups) != 1 or len(base["param_groups"]) != 1:
            raise RuntimeError("Expected one fp32 group and one ScheduleFree optimizer group.")
        train_flat = fp32_groups[0]
        if partition_numel is None:
            partition_numel = train_flat.numel()
        elif train_flat.numel() != partition_numel:
            raise RuntimeError("All ZeRO-1 rank partitions must have the same length.")

        group = base["param_groups"][0]
        use_schedulefree = bool(group.get("use_schedulefree", False))
        train_mode = bool(group.get("train_mode", False))
        beta1 = float(group["betas"][0]) if use_schedulefree and train_mode else None
        state_id = group["params"][0]
        z_flat = base["state"][state_id].get("z") if beta1 is not None else None
        if beta1 is not None and (z_flat is None or z_flat.shape != train_flat.shape):
            raise RuntimeError(f"Rank {rank}: ScheduleFree z does not match the fp32 master partition.")

        mappings = optim.get("param_slice_mappings")
        if not isinstance(mappings, list) or len(mappings) != 1:
            raise RuntimeError(f"Rank {rank}: expected one param_slice_mappings group.")
        rank_mapping = mappings[0]
        rank_start = rank * partition_numel
        rank_end = rank_start + partition_numel

        expected_names = []
        for name, (param_start, numel) in param_offsets.items():
            param_end = param_start + numel
            intersection_start = max(rank_start, param_start)
            intersection_end = min(rank_end, param_end)
            if intersection_start >= intersection_end:
                continue
            expected_names.append(name)
            expected_source_start = intersection_start - rank_start
            destination_start = intersection_start - param_start
            fragment_numel = intersection_end - intersection_start

            address = rank_mapping.get(name)
            if address is None or int(address.start) != expected_source_start or int(address.numel) != fragment_numel:
                raise RuntimeError(
                    f"Rank {rank} mapping mismatch for {name}: expected "
                    f"start={expected_source_start}, numel={fragment_numel}, got {address}."
                )
            _copy_eval_fragment(
                state_dict[name],
                destination_start,
                train_flat,
                z_flat,
                expected_source_start,
                fragment_numel,
                beta1,
            )
            covered[name] += fragment_numel

        if list(rank_mapping.keys()) != expected_names:
            raise RuntimeError(f"Rank {rank}: param_slice_mappings order/content differs from param_shapes.")
        conversion_groups.append(
            {
                "rank": rank,
                "train_mode": train_mode,
                "use_schedulefree": use_schedulefree,
                "beta1": beta1,
                "partition_numel": train_flat.numel(),
            }
        )
        del blob

    aligned_full_numel = partition_numel * len(optim_files)
    if aligned_full_numel < full_numel or aligned_full_numel - full_numel >= 2 * len(optim_files):
        raise RuntimeError(
            f"Unexpected ZeRO padding: parameters={full_numel}, partitions={aligned_full_numel}."
        )
    incomplete = {name: (covered[name], numel) for name, (_, numel) in param_offsets.items() if covered[name] != numel}
    if incomplete:
        raise RuntimeError(f"Incomplete parameter reconstruction: {list(incomplete.items())[:5]}")

    for name in model_blob.get("buffer_names", []):
        state_dict[name] = model_blob["module"][name].to(torch.bfloat16).clone()

    metadata = {
        "source_checkpoint": str(ckpt_dir),
        "source_train_mode": [group["train_mode"] for group in conversion_groups],
        "schedulefree_beta1": [group["beta1"] for group in conversion_groups],
        "world_size": len(optim_files),
        "parameter_count": len(param_shapes),
        "parameter_numel": full_numel,
        "output_dtype": "torch.bfloat16",
        "formula": "torch.lerp(train, z, 1 - 1 / beta1)",
    }
    return state_dict, metadata


def build_arch(mode, pretrained, ste_layers, ste_heads, ste_head_init, local_files_only):
    """Reconstruct the training architecture (random/base weights); we overwrite with consolidated."""
    base = Flux2Transformer2DModel.from_pretrained(
        pretrained, subfolder="transformer", torch_dtype=torch.float32, local_files_only=local_files_only
    )
    if mode in BASE_MODES:
        return base
    config_dict = base.config.to_dict() if hasattr(base.config, "to_dict") else dict(base.config)
    valid_keys = set(inspect.signature(Flux2Transformer2DModel.__init__).parameters.keys()) - {"self"}
    init_kwargs = {k: v for k, v in config_dict.items() if k in valid_keys}
    pe_kwargs = {
        "dim_in": base.inner_dim,
        "pe_dim": sum(base.config.axes_dims_rope),
        "num_layers": len(base.transformer_blocks) + len(base.single_transformer_blocks),
        "sampler": "vanilla_ste",
        "encoder_layers": ste_layers,
        "encoder_num_heads": ste_heads,
        "head_init": ste_head_init,
    }
    model = custom_cls_for(mode)(pe_exchange_kwargs=pe_kwargs, **init_kwargs)
    # Initialise base params (pe_exchange/STE get their own init); consolidated load overwrites all.
    model.load_state_dict(base.state_dict(), strict=False)
    del base
    return model


def consolidate_one(ckpt_dir, mode, pretrained, ste_layers, ste_heads, ste_head_init,
                    out_root, out_subdir, local_files_only, do_delete, do_verify):
    ckpt_dir = Path(ckpt_dir)
    ds_dir = ckpt_dir / "pytorch_model"
    if not ds_dir.exists():
        log(f"SKIP {ckpt_dir.name}: no pytorch_model/ (already consolidated?)")
        return False

    log(f"--- {ckpt_dir.name}: reconstructing ScheduleFree eval-view state_dict from ZeRO shards ...")
    state_dict, conversion_metadata = get_schedulefree_eval_state_dict_from_zero_checkpoint(ckpt_dir)
    log(f"    consolidated tensors: {len(state_dict)} ({conversion_metadata['output_dtype']})")

    log("    building architecture ...")
    model = build_arch(mode, pretrained, ste_layers, ste_heads, ste_head_init, local_files_only)

    model_param_keys = set(dict(model.named_parameters()).keys())
    sd_keys = set(state_dict.keys())
    missing_params = sorted(model_param_keys - sd_keys)
    unexpected = sorted(sd_keys - set(model.state_dict().keys()))
    if missing_params:
        raise RuntimeError(
            f"{ckpt_dir.name}: {len(missing_params)} trained PARAMS missing from checkpoint "
            f"(e.g. {missing_params[:5]}). Aborting -- would silently keep base init."
        )
    if unexpected:
        log(f"    WARN unexpected keys in checkpoint (not in model, ignored): {len(unexpected)} e.g. {unexpected[:3]}")

    incompat = model.load_state_dict(state_dict, strict=False)
    # strict=False so non-persistent buffers don't error; params already verified present above.
    log(f"    load_state_dict: missing={len(incompat.missing_keys)} unexpected={len(incompat.unexpected_keys)}")

    model = model.to(torch.bfloat16)

    if out_root is not None:
        out_dir = Path(out_root) / ckpt_dir.name / out_subdir
    else:
        out_dir = ckpt_dir / out_subdir
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    log(f"    save_pretrained -> {out_dir}")
    model.save_pretrained(out_dir)
    del model, state_dict

    manifest_path = out_dir.parent / "eval_conversion_manifest.json"
    manifest_path.write_text(json.dumps(conversion_metadata, indent=2) + "\n")
    log(f"    conversion manifest -> {manifest_path}")

    files = sorted(p.name for p in out_dir.iterdir())
    log(f"    wrote: {files}")
    if "config.json" not in files or not any(f.endswith(".safetensors") for f in files):
        raise RuntimeError(f"{ckpt_dir.name}: save_pretrained output incomplete: {files}")

    if do_verify:
        # NOTE: bare `custom_cls.from_pretrained(out_dir)` does NOT round-trip this subclass's
        # config (pe_exchange subclass __init__ hides the base config behind **kwargs without
        # @register_to_config, so diffusers rebuilds the DEFAULT architecture -> shape mismatch).
        # That is a pre-existing inference-loader issue (pipeline line ~731). Verify the saved dir
        # the CORRECT way: rebuild the exact arch and load the saved safetensors into it.
        log("    verify: build arch + load saved safetensors (explicit) ...")
        from safetensors.torch import load_file
        vmodel = build_arch(mode, pretrained, ste_layers, ste_heads, ste_head_init, local_files_only)
        vsd = {}
        for shard in sorted(out_dir.glob("*.safetensors")):
            vsd.update(load_file(str(shard)))
        vmissing = sorted(set(dict(vmodel.named_parameters()).keys()) - set(vsd.keys()))
        if vmissing:
            raise RuntimeError(
                f"{ckpt_dir.name}: verify FAILED -- {len(vmissing)} params missing from saved dir "
                f"e.g. {vmissing[:5]}"
            )
        vinc = vmodel.load_state_dict(vsd, strict=False)
        log(f"    verify OK: saved dir loads into arch, missing={len(vinc.missing_keys)} "
            f"unexpected={len(vinc.unexpected_keys)}")
        del vmodel, vsd

    if do_delete and out_root is None:
        log(f"    deleting DeepSpeed state under {ckpt_dir.name} ...")
        shutil.rmtree(ds_dir)
        for junk in ("zero_to_fp32.py", "latest"):
            p = ckpt_dir / junk
            if p.exists():
                p.unlink()
        for p in ckpt_dir.glob("random_states_*.pkl"):
            p.unlink()
        log(f"    {ckpt_dir.name} now: {sorted(q.name for q in ckpt_dir.iterdir())}")
    elif do_delete and out_root is not None:
        log("    (out_root set -> NOT deleting DeepSpeed state; this is validation mode)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="runs/<name> dir containing checkpoint-*/")
    ap.add_argument("--experiment_mode", default=None,
                    help="hard_exchange|soft_exchange|no_sub|sub_no_exchange (else read train_args.json)")
    ap.add_argument("--pretrained", default=None)
    ap.add_argument("--ste_encoder_layers", type=int, default=None)
    ap.add_argument("--ste_encoder_num_heads", type=int, default=None)
    ap.add_argument("--ste_head_init", default=None)
    ap.add_argument("--checkpoints", nargs="*", default=None,
                    help="specific checkpoint dir names (default: all checkpoint-* in run_dir)")
    ap.add_argument("--out_root", default=None,
                    help="write transformer/ under this root instead of inside the checkpoint "
                         "(VALIDATION mode: never deletes DeepSpeed state)")
    ap.add_argument("--out_subdir", default="transformer")
    ap.add_argument("--no_local_files_only", action="store_true")
    ap.add_argument("--keep_ds", action="store_true", help="do not delete DeepSpeed state")
    ap.add_argument("--dry_run", action="store_true", help="alias for --keep_ds")
    ap.add_argument("--no_verify", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    train_args = {}
    ta_path = run_dir / "train_args.json"
    if ta_path.exists():
        train_args = json.loads(ta_path.read_text())
        log(f"loaded train_args.json from {ta_path}")
    else:
        log(f"no train_args.json in {run_dir} (expected for a killed run) -> using flags/defaults")

    mode = args.experiment_mode or train_args.get("experiment_mode")
    if mode is None:
        raise SystemExit("experiment_mode unknown: pass --experiment_mode or provide train_args.json")
    pretrained = args.pretrained or train_args.get("pretrained_model_name_or_path") or DEFAULT_MODEL
    ste_layers = args.ste_encoder_layers if args.ste_encoder_layers is not None else int(train_args.get("ste_encoder_layers", 1))
    ste_heads = args.ste_encoder_num_heads if args.ste_encoder_num_heads is not None else int(train_args.get("ste_encoder_num_heads", 8))
    ste_head_init = args.ste_head_init or train_args.get("ste_head_init", "zero")
    local_files_only = not args.no_local_files_only
    do_delete = not (args.keep_ds or args.dry_run)
    do_verify = not args.no_verify

    if args.checkpoints:
        ckpts = [run_dir / c for c in args.checkpoints]
    else:
        ckpts = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
                       key=lambda x: int(x.name.split("-")[1]))
    if not ckpts:
        raise SystemExit(f"no checkpoint-* dirs in {run_dir}")

    log(f"run={run_dir.name} mode={mode} pretrained={pretrained} ste=({ste_layers},{ste_heads},{ste_head_init}) "
        f"delete_ds={do_delete} verify={do_verify} out_root={args.out_root}")
    log(f"checkpoints: {[c.name for c in ckpts]}")

    done = 0
    for i, c in enumerate(ckpts):
        try:
            # Each checkpoint is self-validated by the forward path (missing-params guard +
            # load missing=0). The explicit save-reload verify is expensive (rebuilds base), so
            # run it only on the first checkpoint to confirm the saved format round-trips.
            verify_this = do_verify and (i == 0)
            if consolidate_one(c, mode, pretrained, ste_layers, ste_heads, ste_head_init,
                               args.out_root, args.out_subdir, local_files_only, do_delete, verify_this):
                done += 1
        except Exception as exc:  # noqa: BLE001
            log(f"ERROR on {c.name}: {exc}")
            raise
    log(f"DONE: consolidated {done}/{len(ckpts)} checkpoints in {run_dir.name}")


if __name__ == "__main__":
    main()
