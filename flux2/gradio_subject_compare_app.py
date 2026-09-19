from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from gradio_compare_worker import model_worker_main


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_COMPARE_OUTPUT_DIR = ROOT_DIR / "gradio_compare_outputs"
DEFAULT_PROMPT_CATALOG = ROOT_DIR / "gradio_test_prompts.json"
NO_MASK_DISABLE_MODE = "No Mask (disable Sub)"
NO_MASK_FULL_SUB_MODE = "No Mask (full-size Sub)"


def load_prompt_catalog(path: Path) -> dict[str, dict[str, Any]]:
    catalog = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(catalog, dict) or not catalog:
        raise ValueError(f"Prompt catalog must be a non-empty JSON object: {path}")
    required = {"slug", "reference_order", "location", "mask_guide", "prompt"}
    for key, entry in catalog.items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            raise ValueError(f"Invalid prompt catalog entry: {key!r}")
        missing = sorted(required - set(entry))
        if missing:
            raise ValueError(f"Prompt catalog entry {key!r} is missing: {missing}")
    return catalog


class DockerStopWatchdog:
    """Continuously stop the two explicitly configured competing containers."""

    def __init__(self, container_ids: list[str], interval_seconds: float = 1.0) -> None:
        self.container_ids = [container_id for container_id in container_ids if container_id]
        self.interval_seconds = float(interval_seconds)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            name="docker-stop-watchdog",
            target=self._run,
            daemon=True,
        )

    def stop_containers(self, *, check: bool) -> None:
        if not self.container_ids:
            return
        result = subprocess.run(
            ["docker", "stop", *self.container_ids],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if check and result.returncode != 0:
            detail = result.stderr.strip() or f"exit code {result.returncode}"
            raise RuntimeError(f"Initial docker stop failed: {detail}")

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self.stop_containers(check=False)

    def start(self) -> None:
        self.stop_containers(check=True)
        print(
            "[docker-watchdog] initial stop complete; repeating every "
            f"{self.interval_seconds:.1f}s for {','.join(self.container_ids)}",
            flush=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


class ManagedModelWorker:
    def __init__(self, context: mp.context.BaseContext, config: dict[str, Any]) -> None:
        self.name = str(config["name"])
        self.request_queue = context.Queue(maxsize=1)
        self.event_queue = context.Queue()
        self.process = context.Process(
            name=f"gradio-{self.name}-worker",
            target=model_worker_main,
            args=(config, self.request_queue, self.event_queue),
            daemon=True,
        )
        self.ready = False
        self.ready_metadata: dict[str, Any] | None = None
        self.total_steps = 50
        self.status_message = "Starting worker"
        self.fatal_error: str | None = None

    def start(self) -> None:
        self.process.start()

    def _next_event(self, timeout: float) -> dict[str, Any]:
        try:
            event = self.event_queue.get(timeout=timeout)
        except queue.Empty:
            if not self.process.is_alive():
                raise RuntimeError(
                    f"{self.name} worker exited before returning a result "
                    f"(exit code {self.process.exitcode})."
                )
            raise

        kind = event.get("kind")
        if kind == "status":
            self.status_message = str(event["message"])
        elif kind == "ready":
            self.ready = True
            self.ready_metadata = event.get("metadata")
            self.total_steps = int(event.get("total_steps", 50))
            self.status_message = "Ready"
        elif kind == "fatal":
            self.fatal_error = str(event.get("message") or "unknown startup error")
        return event

    def wait_until_ready(self, timeout: float = 600.0) -> None:
        deadline = time.monotonic() + timeout
        while not self.ready:
            if self.fatal_error is not None:
                raise RuntimeError(f"{self.name} model failed to load: {self.fatal_error}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for the {self.name} model to load.")
            try:
                self._next_event(min(1.0, remaining))
            except queue.Empty:
                continue

    def submit(self, request_id: str, payload: dict[str, Any]) -> None:
        if not self.ready:
            raise RuntimeError(f"{self.name} model is not ready.")
        self.request_queue.put({"kind": "generate", "request_id": request_id, **payload})

    def next_generation_event(self, timeout: float) -> dict[str, Any]:
        return self._next_event(timeout)

    def close(self) -> None:
        if not self.process.is_alive():
            return
        try:
            self.request_queue.put_nowait({"kind": "shutdown"})
        except queue.Full:
            pass
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)


class ComparisonRuntime:
    def __init__(self, worker_configs: list[dict[str, Any]], output_root: Path) -> None:
        context = mp.get_context("spawn")
        self.workers = {
            config["name"]: ManagedModelWorker(context, config) for config in worker_configs
        }
        self.output_root = Path(output_root)
        self.generation_lock = threading.Lock()
        self._closed = False

    def start(self) -> None:
        for worker in self.workers.values():
            worker.start()

    def wait_until_ready(self, timeout: float = 600.0) -> None:
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            futures = [executor.submit(worker.wait_until_ready, timeout) for worker in self.workers.values()]
            for future in futures:
                future.result()

    def ready_status(self) -> str:
        lines = []
        for name, worker in self.workers.items():
            checkpoint = None
            if worker.ready_metadata:
                checkpoint = worker.ready_metadata.get("checkpoint_dir")
            suffix = f" — {checkpoint}" if checkpoint else ""
            lines.append(f"{name}: {worker.status_message}{suffix}")
        return "\n".join(lines)

    def _prepare_output_bundle(
        self,
        payload: dict[str, Any],
        request_id: str,
    ) -> Path:
        prompt_label = str(payload["prompt_label"])
        timestamp = datetime.now().astimezone()
        output_dir = self.output_root / (
            f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')}_{prompt_label}_{request_id[:8]}"
        )
        output_dir.mkdir(parents=True, exist_ok=False)

        background_path = output_dir / "input_background.png"
        payload["main_image"].convert("RGB").save(background_path)
        subject_paths = []
        for index, subject_image in enumerate(payload["subject_images"], start=1):
            subject_path = output_dir / f"input_subject_{index:02d}.png"
            subject_image.convert("RGB").save(subject_path)
            subject_paths.append(str(subject_path))

        mask_path = None
        if payload["mask"] is not None:
            mask_path = output_dir / "input_mask.png"
            payload["mask"].convert("L").save(mask_path)

        request_record = {
            "request_id": request_id,
            "created_at": timestamp.isoformat(),
            "prompt_label": prompt_label,
            "selected_prompt_key": payload["selected_prompt_key"],
            "user_prompt": payload["user_prompt"],
            "expanded_prompt": payload["prompt"],
            "seed": int(payload["seed"]),
            "guidance_scale": float(payload["guidance_scale"]),
            "safe_cfg_rescale": bool(payload.get("safe_cfg_rescale", False)),
            "guidance_rescale": float(payload.get("guidance_rescale", 0.0)),
            "use_cfg_zero_star": bool(payload.get("use_cfg_zero_star", False)),
            "cfg_zero_star_zero_init_steps": int(
                payload.get("cfg_zero_star_zero_init_steps", 1)
            ),
            "use_subject": bool(payload["use_subject"]),
            "mask_mode": str(payload["mask_mode"]),
            "used_sub_branch": bool(
                payload["mask"] is not None or payload.get("full_sub_without_mask", False)
            ),
            "full_sub_without_mask": bool(payload.get("full_sub_without_mask", False)),
            "include_automatic_prompt_context": bool(
                payload.get("include_automatic_prompt_context", True)
            ),
            "subject_image_count": len(subject_paths),
            "inputs": {
                "background": str(background_path),
                "subjects": subject_paths,
                "mask": str(mask_path) if mask_path is not None else None,
            },
        }
        (output_dir / "request.json").write_text(
            json.dumps(request_record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_dir

    def generate(
        self,
        payload: dict[str, Any],
        progress_callback: Callable[[dict[str, int], dict[str, int]], None],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
        with self.generation_lock:
            self.wait_until_ready()
            request_id = uuid.uuid4().hex
            payload = dict(payload)
            output_dir = self._prepare_output_bundle(payload, request_id)
            payload["output_dir"] = str(output_dir)
            for worker in self.workers.values():
                worker.submit(request_id, payload)

            completed = {name: 0 for name in self.workers}
            totals = {name: worker.total_steps for name, worker in self.workers.items()}
            pending = set(self.workers)
            results: dict[str, dict[str, Any]] = {}
            errors: dict[str, str] = {}
            progress_callback(completed, totals)

            while pending:
                received_event = False
                for name in tuple(pending):
                    worker = self.workers[name]
                    try:
                        event = worker.next_generation_event(timeout=0.1)
                    except queue.Empty:
                        continue
                    received_event = True
                    kind = event.get("kind")
                    event_request_id = event.get("request_id")
                    if kind in {"progress", "result", "error"} and event_request_id != request_id:
                        raise RuntimeError(
                            f"Received a stale {name} event for request {event_request_id!r}."
                        )
                    if kind == "progress":
                        completed[name] = int(event["completed"])
                        totals[name] = int(event["total"])
                        progress_callback(completed, totals)
                    elif kind == "result":
                        completed[name] = totals[name]
                        results[name] = event
                        pending.remove(name)
                        progress_callback(completed, totals)
                    elif kind in {"error", "fatal"}:
                        errors[name] = str(event.get("message") or "unknown worker error")
                        pending.remove(name)
                        progress_callback(completed, totals)
                if not received_event:
                    time.sleep(0.05)

            return results, errors

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for worker in self.workers.values():
            worker.close()


def build_demo(
    runtime: ComparisonRuntime,
    prompt_catalog: dict[str, dict[str, Any]],
    *,
    no_mask_full_sub: bool = False,
    bbox_protocol_compare: bool = False,
):
    import gradio as gr

    from gradio_subject_inputs import (
        MAX_SUBJECT_REFERENCE_IMAGES,
        add_reference_prompt_context,
        classify_test_prompt,
        extract_mask_from_editor,
        load_subject_images,
        sync_editor_background,
    )

    prompt_choices = list(prompt_catalog)
    default_prompt_key = prompt_choices[0]
    default_prompt_entry = prompt_catalog[default_prompt_key]
    no_mask_mode = NO_MASK_FULL_SUB_MODE if no_mask_full_sub else NO_MASK_DISABLE_MODE

    def select_prompt(prompt_key: str):
        entry = prompt_catalog[prompt_key]
        reference_order = " → ".join(entry["reference_order"])
        guide = (
            f"参考图顺序：{reference_order}\n"
            f"Prompt 位置：{entry['location']}\n"
            f"Mask：{entry['mask_guide']}"
        )
        return entry["prompt"], guide

    def run_generation(
        selected_prompt_key: str,
        prompt: str,
        include_automatic_prompt_context: bool,
        main_image,
        use_subject: bool,
        subject_files,
        mask_mode: str,
        uploaded_mask,
        drawn_mask,
        seed: float,
        guidance_scale: float,
        safe_cfg_rescale: bool,
        cfg_rescale_strength: float,
        use_cfg_zero_star: bool,
        cfg_zero_star_zero_init_steps: float,
        progress=gr.Progress(),
    ):
        if main_image is None:
            raise gr.Error("Main image is required.")

        user_prompt = (prompt or "").strip()
        if not user_prompt:
            raise gr.Error("Prompt is required.")

        effective_use_subject = bool(use_subject)
        subject_images = load_subject_images(subject_files) if effective_use_subject else []
        if len(subject_images) > MAX_SUBJECT_REFERENCE_IMAGES:
            raise gr.Error(
                f"Upload at most {MAX_SUBJECT_REFERENCE_IMAGES} subject images "
                "(background + subjects must stay within FLUX.2 Klein's 4-reference limit)."
            )
        selected_entry = prompt_catalog.get(selected_prompt_key)
        prompt_label = (
            str(selected_entry["slug"])
            if selected_entry is not None
            else classify_test_prompt(user_prompt)
        )
        prompt = add_reference_prompt_context(
            user_prompt,
            len(subject_images),
            include_automatic_context=bool(include_automatic_prompt_context),
        )

        mask = None
        if effective_use_subject and mask_mode == "Upload Mask":
            mask = uploaded_mask
        elif effective_use_subject and mask_mode == "Draw On Background":
            mask = extract_mask_from_editor(drawn_mask)
        if effective_use_subject and mask_mode != no_mask_mode and mask is None:
            raise gr.Error("Mask is required. Upload one or draw one on the background image.")
        full_sub_without_mask = bool(
            effective_use_subject and no_mask_full_sub and mask_mode == NO_MASK_FULL_SUB_MODE
        )

        cfg_rescale_strength = float(cfg_rescale_strength)
        if not 0.0 <= cfg_rescale_strength <= 1.0:
            raise gr.Error("CFG rescale strength must be within [0, 1].")
        guidance_rescale = cfg_rescale_strength if bool(safe_cfg_rescale) else 0.0
        cfg_zero_star_zero_init_steps = int(cfg_zero_star_zero_init_steps)
        if not 0 <= cfg_zero_star_zero_init_steps <= 50:
            raise gr.Error("CFG-Zero* zero-init steps must be within [0, 50].")

        payload = {
            "prompt": prompt,
            "user_prompt": user_prompt,
            "prompt_label": prompt_label,
            "selected_prompt_key": selected_prompt_key,
            "main_image": main_image,
            "use_subject": effective_use_subject,
            "mask_mode": mask_mode,
            "subject_images": subject_images,
            "mask": mask,
            "full_sub_without_mask": full_sub_without_mask,
            "seed": int(seed),
            "guidance_scale": float(guidance_scale),
            "safe_cfg_rescale": bool(safe_cfg_rescale),
            "guidance_rescale": guidance_rescale,
            "use_cfg_zero_star": bool(use_cfg_zero_star),
            "cfg_zero_star_zero_init_steps": cfg_zero_star_zero_init_steps,
            "include_automatic_prompt_context": bool(include_automatic_prompt_context),
        }

        def report_progress(completed: dict[str, int], totals: dict[str, int]) -> None:
            done = sum(completed.values())
            total = sum(totals.values())
            description = " | ".join(
                f"{name} {completed[name]}/{totals[name]}" for name in ("Sparse", "BBox")
            )
            progress((done, total), desc=description)

        try:
            results, errors = runtime.generate(payload, report_progress)
        except Exception as exc:  # noqa: BLE001
            raise gr.Error(str(exc)) from exc

        for name, message in errors.items():
            gr.Warning(f"{name} generation failed: {message}")

        sparse = results.get("Sparse", {})
        bbox = results.get("BBox", {})
        sparse_metadata = sparse.get("metadata") or ({"error": errors["Sparse"]} if "Sparse" in errors else None)
        bbox_metadata = bbox.get("metadata") or ({"error": errors["BBox"]} if "BBox" in errors else None)
        return (
            sparse.get("main_image"),
            bbox.get("main_image"),
            sparse.get("sub_image"),
            bbox.get("sub_image"),
            sparse_metadata,
            bbox_metadata,
        )

    with gr.Blocks(title="Subject-Driven FLUX.2 Model Comparison") as demo:
        gr.Markdown(
            "# Subject-Driven FLUX.2 — Sparse vs Dense BBox\n"
            "One input is sent to both GPU models in parallel. Automatic Picture context can be "
            "enabled or disabled. Both models use 50 steps. "
            "Choose a local preset, then upload/draw the target region or use the configured No Mask mode."
        )
        if bbox_protocol_compare:
            gr.Markdown("**Same BBox-trained weights in both lanes.** Left: mask-selected Sub tokens and PE pairs; right: full bbox Sub tokens and PE pairs. Draw/upload a non-rectangular mask to compare cropping; No Mask retains the whole image.")
        status = gr.Textbox(
            label="Model Status",
            value=runtime.ready_status(),
            interactive=False,
            lines=2,
        )
        prompt_choice = gr.Dropdown(
            choices=prompt_choices,
            value=default_prompt_key,
            label="测试 Prompt",
        )
        prompt_guide = gr.Textbox(
            label="参考图与 Mask 提示",
            value=(
                f"参考图顺序：{' → '.join(default_prompt_entry['reference_order'])}\n"
                f"Prompt 位置：{default_prompt_entry['location']}\n"
                f"Mask：{default_prompt_entry['mask_guide']}"
            ),
            interactive=False,
            lines=3,
        )
        prompt = gr.Textbox(
            label="Prompt",
            lines=4,
            value=default_prompt_entry["prompt"],
            placeholder="Describe the subject's placement, pose, viewpoint, and expression...",
        )
        include_automatic_prompt_context = gr.Checkbox(
            label="Automatically add Picture reference context",
            value=True,
        )
        main_image = gr.Image(type="pil", label="Main / Background Image")
        use_subject = gr.Checkbox(label="Use Subject Reference", value=True)
        with gr.Group(visible=True) as subject_group:
            subject_files = gr.File(
                file_count="multiple",
                file_types=["image"],
                type="filepath",
                label="Subject Reference Images (same subject, up to 3)",
            )
        mask_mode = gr.Radio(
            choices=["Upload Mask", "Draw On Background", no_mask_mode],
            value="Draw On Background",
            label="Region Input Mode",
        )
        with gr.Group(visible=False) as upload_mask_group:
            uploaded_mask = gr.Image(type="pil", label="Uploaded Region Image")
        with gr.Group(visible=True) as draw_mask_group:
            drawn_mask = gr.ImageEditor(
                type="pil",
                label="Draw Target Region On Background",
                image_mode="RGBA",
                sources=["upload"],
                transforms=(),
                layers=False,
                brush=gr.Brush(
                    colors=["#ffffff"],
                    color_mode="fixed",
                    default_size=24,
                ),
                eraser=gr.Eraser(default_size=24),
            )
        with gr.Row():
            seed = gr.Number(label="Seed", value=0, precision=0)
            guidance_scale = gr.Slider(
                label="Guidance Scale",
                minimum=1.0,
                maximum=8.0,
                value=4.0,
                step=0.1,
            )
        with gr.Row():
            safe_cfg_rescale = gr.Checkbox(
                label="Safe CFG normalization (CFG Rescale)",
                value=False,
            )
            cfg_rescale_strength = gr.Slider(
                label="CFG Rescale Strength",
                minimum=0.0,
                maximum=1.0,
                value=0.7,
                step=0.05,
            )
        with gr.Row():
            use_cfg_zero_star = gr.Checkbox(
                label="Use CFG-Zero*",
                value=False,
            )
            cfg_zero_star_zero_init_steps = gr.Slider(
                label="CFG-Zero* Zero-init Steps",
                minimum=0,
                maximum=5,
                value=1,
                step=1,
            )
        run_button = gr.Button("Generate Both Models", variant="primary")

        prompt_choice.change(
            fn=select_prompt,
            inputs=prompt_choice,
            outputs=[prompt, prompt_guide],
            show_progress="hidden",
        )

        gr.Markdown("## Main-image comparison")
        with gr.Row():
            sparse_main = gr.Image(type="pil", label="BBox weights / cropped Sub" if bbox_protocol_compare else "Sparse Model")
            bbox_main = gr.Image(type="pil", label="BBox weights / dense Sub" if bbox_protocol_compare else "Dense BBox Model")
        gr.Markdown("## Sub-branch outputs")
        with gr.Row():
            sparse_sub = gr.Image(type="pil", label="Sparse Sub")
            bbox_sub = gr.Image(type="pil", label="Dense BBox Sub")
        with gr.Row():
            sparse_metadata = gr.JSON(label="Sparse Metadata")
            bbox_metadata = gr.JSON(label="Dense BBox Metadata")

        def update_subject_visibility(enabled: bool, mode: str):
            return (
                gr.update(visible=enabled),
                gr.update(visible=enabled),
                gr.update(visible=enabled and mode == "Upload Mask"),
                gr.update(visible=enabled and mode == "Draw On Background"),
            )

        use_subject.change(
            fn=update_subject_visibility,
            inputs=[use_subject, mask_mode],
            outputs=[subject_group, mask_mode, upload_mask_group, draw_mask_group],
            show_progress="hidden",
        )
        mask_mode.change(
            fn=update_subject_visibility,
            inputs=[use_subject, mask_mode],
            outputs=[subject_group, mask_mode, upload_mask_group, draw_mask_group],
            show_progress="hidden",
        )
        main_image.change(
            fn=sync_editor_background,
            inputs=main_image,
            outputs=drawn_mask,
            show_progress="hidden",
        )
        run_button.click(
            fn=run_generation,
            inputs=[
                prompt_choice,
                prompt,
                include_automatic_prompt_context,
                main_image,
                use_subject,
                subject_files,
                mask_mode,
                uploaded_mask,
                drawn_mask,
                seed,
                guidance_scale,
                safe_cfg_rescale,
                cfg_rescale_strength,
                use_cfg_zero_star,
                cfg_zero_star_zero_init_steps,
            ],
            outputs=[
                sparse_main,
                bbox_main,
                sparse_sub,
                bbox_sub,
                sparse_metadata,
                bbox_metadata,
            ],
        )
    return demo.queue(default_concurrency_limit=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="One Gradio page comparing sparse and dense-bbox models.")
    parser.add_argument("--pretrained-model-name-or-path", type=Path, required=True)
    parser.add_argument("--sparse-checkpoint-path", type=Path, required=True)
    parser.add_argument("--bbox-checkpoint-path", type=Path, required=True)
    parser.add_argument("--sparse-ste-mask-output-dir", type=Path, required=True)
    parser.add_argument("--bbox-ste-mask-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_COMPARE_OUTPUT_DIR)
    parser.add_argument("--prompt-catalog", type=Path, default=DEFAULT_PROMPT_CATALOG)
    parser.add_argument("--sparse-gpu", default="4")
    parser.add_argument("--bbox-gpu", default="5")
    parser.add_argument("--sparse-docker-stop-id", default="")
    parser.add_argument("--bbox-docker-stop-id", default="")
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no-mask-full-sub", action="store_true")
    args = parser.parse_args()
    prompt_catalog = load_prompt_catalog(args.prompt_catalog)

    docker_watchdog = DockerStopWatchdog(
        [args.sparse_docker_stop_id, args.bbox_docker_stop_id],
        interval_seconds=1.0,
    )
    runtime = ComparisonRuntime(
        [
            {
                "name": "Sparse",
                "variant": "sparse",
                "physical_gpu": args.sparse_gpu,
                "checkpoint_path": str(args.sparse_checkpoint_path),
                "pretrained_model_name_or_path": str(args.pretrained_model_name_or_path),
                "ste_mask_output_dir": str(args.sparse_ste_mask_output_dir),
                "local_files_only": args.local_files_only,
            },
            {
                "name": "BBox",
                "variant": "bbox",
                "physical_gpu": args.bbox_gpu,
                "checkpoint_path": str(args.bbox_checkpoint_path),
                "pretrained_model_name_or_path": str(args.pretrained_model_name_or_path),
                "ste_mask_output_dir": str(args.bbox_ste_mask_output_dir),
                "local_files_only": args.local_files_only,
            },
        ],
        output_root=args.output_dir,
    )
    docker_watchdog.start()

    try:
        runtime.start()
        runtime.wait_until_ready()
        demo = build_demo(runtime, prompt_catalog, no_mask_full_sub=args.no_mask_full_sub)
        demo.launch(
            server_name=args.server_name,
            server_port=args.server_port,
            share=args.share,
            allowed_paths=[
                str(args.output_dir.resolve()),
                str(args.sparse_ste_mask_output_dir.resolve()),
                str(args.bbox_ste_mask_output_dir.resolve()),
            ],
        )
    finally:
        runtime.close()
        docker_watchdog.close()


if __name__ == "__main__":
    main()
