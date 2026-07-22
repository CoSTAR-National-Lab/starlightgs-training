# Third-Party Licenses

This project (and the Docker image built from it) bundles third-party software. Each
component remains under its own licence, listed below. **The most restrictive term
governs the combined work: the 3D Gaussian Splatting rasterizer is licensed for
non-commercial research/evaluation only (see the Inria licence).**

| Component | Source | Licence | Commercial use |
|---|---|---|---|
| **diff-gaussian-rasterization** | Inria / MPII (via graphdeco-inria/gaussian-splatting) | Gaussian-Splatting License | ❌ Non-commercial only; requires Inria consent for commercial use |
| **simple-knn** | Inria / MPII (via graphdeco-inria/gaussian-splatting) | Gaussian-Splatting License | ❌ Non-commercial only |
| **nvdiffrast** v0.4.0 | NVIDIA (NVlabs/nvdiffrast) | NVIDIA Source Code License (1-Way Commercial) | ✅ Permitted, with conditions |
| **GS-IR** | lzhnb/GS-IR | See upstream repository | Inherits the non-commercial constraint via the rasterizer |
| **COLMAP** | colmap.github.io | BSD 3-Clause | ✅ Permissive |
| **NVIDIA PyTorch base image** (`nvcr.io/nvidia/pytorch:25.03-py3`) | NVIDIA NGC | NVIDIA Deep Learning Container Licence | ✅ Redistribution permitted under its terms |
| **Python packages** (plyfile, tqdm, trimesh, imageio, kornia, lpips) | PyPI | MIT / BSD / Apache-2.0 (per package) | ✅ Permissive |
| **ffmpeg** | ffmpeg.org (Ubuntu package) | LGPL-2.1+/GPL (depending on build) | ✅ (used as an unmodified tool) |

## Where to find the full licence texts

- **Inria Gaussian-Splatting License**: `submodules/diff-gaussian-rasterization/LICENSE.md`
  (and the identical terms for `simple-knn`). Key clause: *"The Software may be used
  'non-commercially', i.e., for research and/or evaluation purposes only … THE USER CANNOT
  USE, EXPLOIT OR DISTRIBUTE THE SOFTWARE FOR COMMERCIAL PURPOSES WITHOUT PRIOR AND EXPLICIT
  CONSENT OF LICENSORS. YOU MUST CONTACT INRIA FOR ANY [commercial use]."*
- **nvdiffrast**: https://github.com/NVlabs/nvdiffrast/blob/v0.4.0/LICENSE.txt
- **COLMAP**, **NVIDIA base image**, and the **Python packages**, under their respective
  upstream projects / package metadata.
