# Release checks (2026-09-19)

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
