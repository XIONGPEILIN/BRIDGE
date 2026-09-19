# BBox weights: cropped Sub support with visible mask boundaries

Below are six existing custom-input comparisons, all at **50 steps, seed=0,
no latent blending**. CFG is shown for each case (2 or 4). Both columns use the
same BBox-trained checkpoint. **Red lines mark the submitted mask boundary**,
mapped onto Main/Sub outputs; they are visualization overlays, not generated
object outlines. These are the original saved outline PNGs, not newly redrawn
images. Backgrounds, subject references and input masks are shown alongside
every case. Internal-dataset examples are excluded.

Both sub-token support and PE candidate support change from bbox-wide to mask-only;
this is not a token-count-only ablation. Main generates the full image.

#### Example 1: 1092 → 512 Sub tokens (CFG=4)

[Prompt and parameters](20260910_055214_260277_upload_custom_b62404d86a8d_d2bb1844/metadata.json) · 50 steps · seed 0 · same BBox-trained weights

| Input background | Subject reference 1 | Subject reference 2 | Input mask |
|---|---|---|---|
| ![Background](20260910_055214_260277_upload_custom_b62404d86a8d_d2bb1844/input_background.png) | ![Subject reference 1](20260910_055214_260277_upload_custom_b62404d86a8d_d2bb1844/input_subject_01.png) | ![Subject reference 2](20260910_055214_260277_upload_custom_b62404d86a8d_d2bb1844/input_subject_02.png) | ![Submitted mask](20260910_055214_260277_upload_custom_b62404d86a8d_d2bb1844/input_mask.png) |

| Cropped/sparse Sub: Main output | Full bbox Sub: Main output |
|---|---|
| ![Cropped main with red mask boundary](20260910_055214_260277_upload_custom_b62404d86a8d_d2bb1844/sparse_main_mask_outline.png) | ![Dense main with red mask boundary](20260910_055214_260277_upload_custom_b62404d86a8d_d2bb1844/dense_main_mask_outline.png) |

| Cropped/sparse Sub output | Full bbox Sub output |
|---|---|
| ![Cropped sub with red mask boundary](20260910_055214_260277_upload_custom_b62404d86a8d_d2bb1844/sparse_sub_mask_outline.png) | ![Dense sub with red mask boundary](20260910_055214_260277_upload_custom_b62404d86a8d_d2bb1844/dense_sub_mask_outline.png) |

#### Example 2: 300 → 222 Sub tokens (CFG=4)

[Prompt and parameters](20260910_062504_041200_upload_custom_b62404d86a8d_64eb2886/metadata.json) · 50 steps · seed 0 · same BBox-trained weights

| Input background | Subject reference 1 | Subject reference 2 | Input mask |
|---|---|---|---|
| ![Background](20260910_062504_041200_upload_custom_b62404d86a8d_64eb2886/input_background.png) | ![Subject reference 1](20260910_062504_041200_upload_custom_b62404d86a8d_64eb2886/input_subject_01.png) | ![Subject reference 2](20260910_062504_041200_upload_custom_b62404d86a8d_64eb2886/input_subject_02.png) | ![Submitted mask](20260910_062504_041200_upload_custom_b62404d86a8d_64eb2886/input_mask.png) |

| Cropped/sparse Sub: Main output | Full bbox Sub: Main output |
|---|---|
| ![Cropped main with red mask boundary](20260910_062504_041200_upload_custom_b62404d86a8d_64eb2886/sparse_main_mask_outline.png) | ![Dense main with red mask boundary](20260910_062504_041200_upload_custom_b62404d86a8d_64eb2886/dense_main_mask_outline.png) |

| Cropped/sparse Sub output | Full bbox Sub output |
|---|---|
| ![Cropped sub with red mask boundary](20260910_062504_041200_upload_custom_b62404d86a8d_64eb2886/sparse_sub_mask_outline.png) | ![Dense sub with red mask boundary](20260910_062504_041200_upload_custom_b62404d86a8d_64eb2886/dense_sub_mask_outline.png) |

#### Example 3: 1178 → 677 Sub tokens (CFG=4)

[Prompt and parameters](20260910_152028_209813_upload_custom_b62404d86a8d_77f7824a/metadata.json) · 50 steps · seed 0 · same BBox-trained weights

| Input background | Subject reference 1 | Subject reference 2 | Input mask |
|---|---|---|---|
| ![Background](20260910_152028_209813_upload_custom_b62404d86a8d_77f7824a/input_background.png) | ![Subject reference 1](20260910_152028_209813_upload_custom_b62404d86a8d_77f7824a/input_subject_01.png) | ![Subject reference 2](20260910_152028_209813_upload_custom_b62404d86a8d_77f7824a/input_subject_02.png) | ![Submitted mask](20260910_152028_209813_upload_custom_b62404d86a8d_77f7824a/input_mask.png) |

| Cropped/sparse Sub: Main output | Full bbox Sub: Main output |
|---|---|
| ![Cropped main with red mask boundary](20260910_152028_209813_upload_custom_b62404d86a8d_77f7824a/sparse_main_mask_outline.png) | ![Dense main with red mask boundary](20260910_152028_209813_upload_custom_b62404d86a8d_77f7824a/dense_main_mask_outline.png) |

| Cropped/sparse Sub output | Full bbox Sub output |
|---|---|
| ![Cropped sub with red mask boundary](20260910_152028_209813_upload_custom_b62404d86a8d_77f7824a/sparse_sub_mask_outline.png) | ![Dense sub with red mask boundary](20260910_152028_209813_upload_custom_b62404d86a8d_77f7824a/dense_sub_mask_outline.png) |

#### Example 4: 713 → 387 Sub tokens (CFG=4)

[Prompt and parameters](20260910_170030_673036_upload_custom_b62404d86a8d_93c533f0/metadata.json) · 50 steps · seed 0 · same BBox-trained weights

| Input background | Subject reference 1 | Subject reference 2 | Input mask |
|---|---|---|---|
| ![Background](20260910_170030_673036_upload_custom_b62404d86a8d_93c533f0/input_background.png) | ![Subject reference 1](20260910_170030_673036_upload_custom_b62404d86a8d_93c533f0/input_subject_01.png) | ![Subject reference 2](20260910_170030_673036_upload_custom_b62404d86a8d_93c533f0/input_subject_02.png) | ![Submitted mask](20260910_170030_673036_upload_custom_b62404d86a8d_93c533f0/input_mask.png) |

| Cropped/sparse Sub: Main output | Full bbox Sub: Main output |
|---|---|
| ![Cropped main with red mask boundary](20260910_170030_673036_upload_custom_b62404d86a8d_93c533f0/sparse_main_mask_outline.png) | ![Dense main with red mask boundary](20260910_170030_673036_upload_custom_b62404d86a8d_93c533f0/dense_main_mask_outline.png) |

| Cropped/sparse Sub output | Full bbox Sub output |
|---|---|
| ![Cropped sub with red mask boundary](20260910_170030_673036_upload_custom_b62404d86a8d_93c533f0/sparse_sub_mask_outline.png) | ![Dense sub with red mask boundary](20260910_170030_673036_upload_custom_b62404d86a8d_93c533f0/dense_sub_mask_outline.png) |

#### Example 5: 1302 → 618 Sub tokens (CFG=2)

[Prompt and parameters](bbox_weights_sparse_20260910_a/metadata.json) · 50 steps · seed 0 · same BBox-trained weights

| Input background | Subject reference 1 | Subject reference 2 | Input mask |
|---|---|---|---|
| ![Background](bbox_weights_sparse_20260910_a/input_background.png) | ![Subject reference 1](bbox_weights_sparse_20260910_a/input_subject_01.png) | ![Subject reference 2](bbox_weights_sparse_20260910_a/input_subject_02.png) | ![Submitted mask](bbox_weights_sparse_20260910_a/input_mask.png) |

| Cropped/sparse Sub: Main output | Full bbox Sub: Main output |
|---|---|
| ![Cropped main with red mask boundary](bbox_weights_sparse_20260910_a/sparse_main_mask_outline.png) | ![Dense main with red mask boundary](bbox_weights_sparse_20260910_a/dense_main_mask_outline.png) |

| Cropped/sparse Sub output | Full bbox Sub output |
|---|---|
| ![Cropped sub with red mask boundary](bbox_weights_sparse_20260910_a/sparse_sub_mask_outline.png) | ![Dense sub with red mask boundary](bbox_weights_sparse_20260910_a/dense_sub_mask_outline.png) |

#### Example 6: 1302 → 618 Sub tokens (CFG=4)

[Prompt and parameters](bbox_weights_sparse_cfg4_20260910_a/metadata.json) · 50 steps · seed 0 · same BBox-trained weights

| Input background | Subject reference 1 | Subject reference 2 | Input mask |
|---|---|---|---|
| ![Background](bbox_weights_sparse_cfg4_20260910_a/input_background.png) | ![Subject reference 1](bbox_weights_sparse_cfg4_20260910_a/input_subject_01.png) | ![Subject reference 2](bbox_weights_sparse_cfg4_20260910_a/input_subject_02.png) | ![Submitted mask](bbox_weights_sparse_cfg4_20260910_a/input_mask.png) |

| Cropped/sparse Sub: Main output | Full bbox Sub: Main output |
|---|---|
| ![Cropped main with red mask boundary](bbox_weights_sparse_cfg4_20260910_a/sparse_main_mask_outline.png) | ![Dense main with red mask boundary](bbox_weights_sparse_cfg4_20260910_a/dense_main_mask_outline.png) |

| Cropped/sparse Sub output | Full bbox Sub output |
|---|---|
| ![Cropped sub with red mask boundary](bbox_weights_sparse_cfg4_20260910_a/sparse_sub_mask_outline.png) | ![Dense sub with red mask boundary](bbox_weights_sparse_cfg4_20260910_a/dense_sub_mask_outline.png) |
