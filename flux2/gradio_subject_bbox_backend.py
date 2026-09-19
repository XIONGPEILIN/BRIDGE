from __future__ import annotations

from pathlib import Path
from typing import Any

from gradio_subject_backend import REPO_ROOT, SubjectDrivenGradioBackend


DEFAULT_BBOX_RUN_DIR = (
    REPO_ROOT / "runs" / "flux2_klein_ablation_hard_exchange_dense_bbox_pe_bbox"
)


class BBoxSubjectDrivenGradioBackend(SubjectDrivenGradioBackend):
    """Gradio backend matching dense-bbox sub training with bbox-wide PE exchange."""

    def __init__(
        self,
        *,
        run_dir: str | Path = DEFAULT_BBOX_RUN_DIR,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            run_dir=run_dir,
            sub_region_mode="bbox",
            use_sparse_sub_branch=False,
            pe_exchange_region="bbox",
            **kwargs,
        )
