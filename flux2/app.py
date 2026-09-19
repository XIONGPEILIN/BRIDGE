"""Portable launcher for the existing two-worker BRIDGE comparison UI."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, required=True,
                        help="Local hf download directory containing flux2-klein-9b/.")
    parser.add_argument("--base-model", default="black-forest-labs/FLUX.2-klein-base-9B")
    parser.add_argument("--sparse-gpu", default="0")
    parser.add_argument("--bbox-gpu", default="1")
    parser.add_argument("--bbox-protocol-compare", action="store_true",
                        help="Load BBox weights in both lanes; compare mask-sparse versus dense-bbox inference.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/flux2"))
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.sparse_gpu == args.bbox_gpu:
        parser.error("Use two distinct GPUs for the two simultaneously resident models.")
    root = args.weights_dir.resolve() / "flux2-klein-9b"
    checkpoints = {
        "sparse": root / "sparse-mask" / "checkpoint-1400",
        "bbox": root / "dense-bbox" / "checkpoint-1400",
    }
    if args.bbox_protocol_compare:
        checkpoints['sparse'] = checkpoints['bbox']
    for checkpoint in checkpoints.values():
        for name in ("eval_conversion_manifest.json", "transformer/config.json",
                     "transformer/diffusion_pytorch_model.safetensors.index.json"):
            if not (checkpoint / name).is_file():
                parser.error(f"Incomplete checkpoint: {checkpoint / name}")

    from gradio_subject_compare_app import ComparisonRuntime, build_demo, load_prompt_catalog

    output = args.output_dir.resolve()
    configs = [
        {"name": name, "variant": variant, "physical_gpu": gpu,
         "checkpoint_path": str(checkpoints[variant]),
         "pretrained_model_name_or_path": args.base_model,
         "ste_mask_output_dir": str(output / f"{variant}_ste_masks"),
         "local_files_only": args.local_files_only}
        for name, variant, gpu in (("Sparse", "sparse", args.sparse_gpu), ("BBox", "bbox", args.bbox_gpu))
    ]
    runtime = ComparisonRuntime(configs, output_root=output)
    prompts = load_prompt_catalog(Path(__file__).with_name("gradio_test_prompts.json"))
    # No Docker watchdog, host-specific mounts, automatic service changes, or public share link.
    try:
        runtime.start()
        runtime.wait_until_ready()
        demo = build_demo(runtime, prompts, no_mask_full_sub=True,
                          bbox_protocol_compare=args.bbox_protocol_compare)
        demo.launch(server_name=args.server_name, server_port=args.server_port,
                    allowed_paths=[str(output)])
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
