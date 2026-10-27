# Third-party notices

BioSphere's own source code is under the [MIT License](LICENSE). It depends on, downloads or
calls other software and model weights that keep **their own licences**. None of the model
weights are stored in this repository; they are fetched at runtime, and only when you allow
it (`BIOSPHERE_AI_ALLOW_DOWNLOADS=true`) or install them yourself.

**Check these licences before any commercial use.** Some of the weights below are licensed for
non-commercial use only.

## Models (optional features)

| Used for | Model | Licence (as published by its authors) |
|---|---|---|
| Furniture removal — fill | LaMa, `big-lama` weights | Code Apache-2.0; the `big-lama` weights are released under **CC BY-NC-SA 4.0 (non-commercial)** |
| Furniture removal / staging — inpainting | `stable-diffusion-v1-5/stable-diffusion-inpainting` | CreativeML OpenRAIL-M |
| Staging — structure guidance | `lllyasviel/control_v11f1p_sd15_depth`, `lllyasviel/control_v11p_sd15_canny` | OpenRAIL (see model cards) |
| Staging — depth | `depth-anything/Depth-Anything-V2-Small-hf` | Apache-2.0 (the Base/Large variants are non-commercial) |
| Room segmentation | `facebook/mask2former-swin-large-ade-semantic` | Mask2Former code MIT; check the model card for the weights |
| 3D walkthrough — loop-closure retrieval | DINOv2 ViT-S/14 (`facebookresearch/dinov2`, torch.hub) | Apache-2.0 |
| 3D walkthrough — dense start (off by default) | Depth Anything 3 | Varies by model size; the larger models are **non-commercial** |
| 360 studio mode — parallax repair | RAFT (torchvision `raft_small`) | BSD-3-Clause |
| Listing copy, investment assistant | Anthropic Claude (hosted API) | Commercial service under Anthropic's terms; requires your own API key |

## Tools and libraries

| Component | Licence |
|---|---|
| FFmpeg (external binary) | LGPL-2.1+ or GPL, depending on the build you install |
| OpenCV (`opencv-contrib-python`) | Apache-2.0 |
| NumPy, SciPy, scikit-image | BSD-3-Clause |
| FastAPI, Pydantic, SQLModel | MIT |
| PyTorch, torchvision | BSD-3-Clause |
| COLMAP / pycolmap, GLOMAP | BSD-3-Clause |
| LightGlue | Apache-2.0 |
| ALIKED | BSD-3-Clause |
| kornia | Apache-2.0 |
| Brush (3DGS trainer, external binary) | Apache-2.0 |
| `@playcanvas/splat-transform`, PlayCanvas engine | MIT |
| three.js, @react-three/fiber, @react-three/drei | MIT |
| Next.js, React | MIT |

This list covers the components that matter for licensing decisions; the full dependency
trees are in `engine/uv.lock` and `web/package-lock.json`. Licences can change between
versions — the authors' own pages are authoritative.
