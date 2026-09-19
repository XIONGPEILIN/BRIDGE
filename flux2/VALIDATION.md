# Release checks (2026-09-19)

## Subject-condition and gallery extension

- Exact 27,834/3,092 train/test ordering, prompts and retained non-path values
  checked against source manifests. All non-target image paths resolve to either
  original HF assets or explicitly inventoried archive members.
- 30,926 generated subject conditions exist; generation records identify
  `Qwen/Qwen-Image-Edit-2511`. Archives contain 54,413 assets / 31,403,314,841
  payload bytes including missing crop/background/mask dependencies.
- Six custom-input same-BBox-weight comparisons included; all three internal
  dataset cases excluded. Original images copied without pixel modification.
- Training/cache/bucketing/converter/preparation/launcher `--help` all passed;
  training shell syntax passed. **Nine CPU tests passed**, including checked
  archive extraction, absolute-path preparation and refusal to overwrite
  different existing files.
- Same-weight Gradio UI builds with **43 components** and explicit labels.
  Public launcher routes both arms to BBox weights while retaining independent
  sparse and dense layouts. This option has not been GPU-replayed during release.
- Paper method figure is copied unchanged. Qwen LoRA versus FLUX full training
  is documented as an implementation difference of the same method.
- No new training, cache generation or image inference was launched.

## Initial model release checks

- Both step-1400 eval exports: all safetensors shard indexes, offsets, tensor
  sizes and BF16 dtypes checked; full weight-file SHA-256 checksums computed.
- Both exports match the custom transformer's complete **714 state-tensor keys
  and shapes** using metadata-only construction: 713 parameter tensors plus
  the scalar STE-temperature buffer, 9,098,549,280 parameter elements total.
- **8 CPU unit tests passed**: full-size sub construction, independent Gaussian
  sub initialization, removal of an interior sparse token at fixed bbox,
  dense-bbox retention of interior holes, and STE padding behavior.
- The Gradio comparison UI constructed successfully with **42 components**, using
  a dummy runtime: no service, GPU model load, or new image generation was started.
- Portable launcher's `--help` and all packaged Python syntax checked.
- The four example output images were copied byte-for-byte from the existing
  2026-09-04 result. Their source records identify the two eval exports, no missing
  or unexpected weight keys, and 50 inference steps. Public metadata includes
  image and source-metadata SHA-256 hashes.

Checks ran in the project environment using the matching vendored Diffusers
commit via `PYTHONPATH`; the old editable install was not portable outside the
project checkout. The release requirements point to the same public commit.
A fresh dependency installation and a new full 50-step GPU inference run were
**not** performed. Historical example images retain their CUDA-RNG provenance.
