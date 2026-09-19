from __future__ import annotations

from pathlib import Path

from PIL import Image


PROMPT_PREFIX = "Picture 1 is the image to modify."
MAX_SUBJECT_REFERENCE_IMAGES = 3
TEST_PROMPT_MARKERS = (
    ("08_comprehensive", "shown in a left-facing three-quarter side view while looking back"),
    ("07_angry", "with an angry expression"),
    ("06_sad", "with a clearly sad expression"),
    ("05_happy", "with a joyful expression"),
    ("04_standing_on_water", "shown in a three-quarter front view"),
    ("03_back_view", "in a clear full-body back view"),
    ("02_side_view", "in a clear full-body left-facing side profile"),
    ("01_front_view", "in a clear full-body front view"),
)


def classify_test_prompt(prompt: str) -> str:
    normalized = " ".join((prompt or "").lower().split())
    for label, marker in TEST_PROMPT_MARKERS:
        if marker in normalized:
            return label
    return "custom"


def load_subject_images(subject_files) -> list[Image.Image]:
    images: list[Image.Image] = []
    for subject_file in subject_files or []:
        file_path = subject_file if isinstance(subject_file, (str, Path)) else subject_file.name
        with Image.open(file_path) as image:
            images.append(image.convert("RGB").copy())
    return images


def add_reference_prompt_context(
    prompt: str,
    subject_count: int,
    *,
    include_automatic_context: bool = True,
) -> str:
    if not include_automatic_context:
        return prompt
    if not prompt.lower().startswith(PROMPT_PREFIX.lower()):
        prompt = f"{PROMPT_PREFIX} {prompt}"
    if subject_count == 1:
        return f"{prompt} Picture 2 is a reference image of the subject."
    if subject_count > 1:
        last_picture = subject_count + 1
        return (
            f"{prompt} Pictures 2 through {last_picture} are reference images of the same subject "
            "from different views."
        )
    return prompt


def sync_editor_background(main_image):
    if main_image is None:
        return None
    return {"background": main_image, "layers": [], "composite": main_image}


def extract_mask_from_editor(editor_value) -> Image.Image | None:
    if not editor_value:
        return None

    layers = editor_value.get("layers") or []
    background = editor_value.get("background")
    if background is None:
        return None

    if not layers:
        width, height = background.size
        return Image.new("L", (width, height), 0)

    alpha = Image.new("L", background.size, 0)
    for layer in layers:
        rgba = layer.convert("RGBA")
        layer_alpha = rgba.getchannel("A")
        alpha = Image.composite(
            Image.new("L", background.size, 255),
            alpha,
            layer_alpha,
        )
    return alpha.point(lambda x: 255 if x > 0 else 0, mode="L")
