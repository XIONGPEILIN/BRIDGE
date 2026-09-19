from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def _emit(event_queue, worker_name: str, kind: str, **payload: Any) -> None:
    event_queue.put({"worker": worker_name, "kind": kind, **payload})


def _save_generation_result(
    *,
    worker_name: str,
    variant: str,
    request_id: str,
    request: dict[str, Any],
    result: Any,
) -> tuple[str, str | None, dict[str, Any]]:
    output_dir = Path(request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = variant.lower()

    main_path = output_dir / f"{slug}_main.png"
    result.main_image.save(main_path)
    sub_path = None
    if result.sub_image is not None:
        sub_path = output_dir / f"{slug}_sub.png"
        result.sub_image.save(sub_path)

    metadata = {
        **result.metadata,
        "request_id": request_id,
        "worker": worker_name,
        "variant": variant,
        "prompt_label": request["prompt_label"],
        "selected_prompt_key": request["selected_prompt_key"],
        "saved_output_dir": str(output_dir),
        "saved_main_image": str(main_path),
        "saved_sub_image": str(sub_path) if sub_path is not None else None,
    }
    metadata_path = output_dir / f"{slug}_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(
        f"[compare-worker:{worker_name}] RESULT_SAVED "
        f"label={request['prompt_label']} dir={output_dir}",
        flush=True,
    )
    return str(main_path), str(sub_path) if sub_path is not None else None, metadata


def model_worker_main(config: dict[str, Any], request_queue, event_queue) -> None:
    """Own one model and one visible GPU for the lifetime of the comparison UI."""
    worker_name = str(config["name"])
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config["physical_gpu"])
        _emit(event_queue, worker_name, "status", message="Loading model")

        if config["variant"] == "bbox":
            from gradio_subject_bbox_backend import BBoxSubjectDrivenGradioBackend

            backend_cls = BBoxSubjectDrivenGradioBackend
        elif config["variant"] == "sparse":
            from gradio_subject_backend import SubjectDrivenGradioBackend

            backend_cls = SubjectDrivenGradioBackend
        else:
            raise ValueError(f"Unknown worker variant: {config['variant']!r}")

        backend = backend_cls(
            checkpoint_path=config["checkpoint_path"],
            pretrained_model_name_or_path=config["pretrained_model_name_or_path"],
            device="cuda:0",
            local_files_only=bool(config["local_files_only"]),
            ste_mask_output_dir=config["ste_mask_output_dir"],
        )
        print(
            f"[compare-worker:{worker_name}] MODEL_READY "
            f"checkpoint={config['checkpoint_path']} "
            f"base_transformer_loaded={backend.load_metadata.get('base_transformer_loaded')}",
            flush=True,
        )
        _emit(
            event_queue,
            worker_name,
            "ready",
            metadata=backend.load_metadata,
            total_steps=backend.num_inference_steps,
        )

        while True:
            request = request_queue.get()
            if request.get("kind") == "shutdown":
                return
            if request.get("kind") != "generate":
                raise ValueError(f"Unknown worker request: {request.get('kind')!r}")

            request_id = str(request["request_id"])
            try:
                result = backend.generate(
                    prompt=request["prompt"],
                    main_image=request["main_image"],
                    use_subject=bool(request["use_subject"]),
                    subject_image=request["subject_images"],
                    mask=request["mask"],
                    full_sub_without_mask=bool(request.get("full_sub_without_mask", False)),
                    seed=int(request["seed"]),
                    guidance_scale=float(request["guidance_scale"]),
                    guidance_rescale=float(request.get("guidance_rescale", 0.0)),
                    use_cfg_zero_star=bool(request.get("use_cfg_zero_star", False)),
                    cfg_zero_star_zero_init_steps=int(
                        request.get("cfg_zero_star_zero_init_steps", 1)
                    ),
                    progress_callback=lambda completed, total: _emit(
                        event_queue,
                        worker_name,
                        "progress",
                        request_id=request_id,
                        completed=int(completed),
                        total=int(total),
                    ),
                )
                main_image_path, sub_image_path, metadata = _save_generation_result(
                    worker_name=worker_name,
                    variant=str(config["variant"]),
                    request_id=request_id,
                    request=request,
                    result=result,
                )
                _emit(
                    event_queue,
                    worker_name,
                    "result",
                    request_id=request_id,
                    main_image=main_image_path,
                    sub_image=sub_image_path,
                    metadata=metadata,
                )
            except Exception as exc:  # noqa: BLE001
                trace = traceback.format_exc()
                print(
                    f"[compare-worker:{worker_name}] GENERATION_ERROR: {exc}\n{trace}",
                    file=sys.stderr,
                    flush=True,
                )
                _emit(
                    event_queue,
                    worker_name,
                    "error",
                    request_id=request_id,
                    message=str(exc),
                    traceback=trace,
                )
    except BaseException as exc:  # noqa: BLE001
        trace = traceback.format_exc()
        print(f"[compare-worker:{worker_name}] FATAL: {exc}\n{trace}", file=sys.stderr, flush=True)
        _emit(
            event_queue,
            worker_name,
            "fatal",
            message=str(exc),
            traceback=trace,
        )
        raise
