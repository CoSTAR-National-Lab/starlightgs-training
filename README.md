# GS-IR (CoSTAR)

A containerised fork of [GS-IR](https://github.com/lzhnb/GS-IR) (3D Gaussian Splatting for
Inverse Rendering). You give it photos or a video of a scene, and it reconstructs the geometry
and physically based materials (albedo, roughness, metallic, normals) and exports a material
PLY. Everything runs from a single Docker command, with no Python or manual setup.

> **Non-commercial use only.** This project bundles the Inria 3D Gaussian Splatting rasterizer,
> which is licensed for research and evaluation purposes only. Do not use it, or its outputs,
> for commercial work without the appropriate licence. See
> [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## What this fork adds to GS-IR

* A one command Docker pipeline that runs the whole flow (calibrate, train, bake, decompose,
  render, export) unattended. No conda, no compiling.
* Automatic camera calibration. You can give it raw photos or a video and it runs COLMAP for you
  the first time, so you do not have to prepare the data yourself.
* A single image that runs across NVIDIA Turing to Hopper GPUs (RTX 20, 30 and 40 series, A100,
  H100 and H200).
* A material PLY exporter (`extract_ply.py`).
* A NeRF-OSR scene reader (`scene/nerfosr_reader.py`) for training on NeRF-OSR ground truth pose
  datasets. NeRF-OSR data:
  [download](https://nextcloud.mpi-klsb.mpg.de/index.php/s/mGXYKpD8raQ8nMk),
  [project and code](https://github.com/r00tman/NeRF-OSR).

## Requirements

* An NVIDIA GPU (2018 or newer) with a recent driver, and Docker.
* Linux, or Windows 10 or 11 with WSL2 and Docker Desktop.
* Apple Silicon and Intel Macs are not supported (no CUDA), and neither are AMD GPUs.

Full hardware notes and troubleshooting are in
[packaging/README_USERS.md](packaging/README_USERS.md).

## Quick start

Get the image:

```bash
docker pull ghcr.io/costar/gs-ir:latest
```

(or load a shared `.tar` file, see the user guide).

You also need the launcher script for your operating system. It is a small wrapper that fills in
the Docker options for you, so copy it onto your machine from [packaging/](packaging/):

* Linux, WSL or a macOS shell: `run.sh`
* Windows: `run.bat`

Then run it on a scene folder (a video, loose photos, or an already calibrated dataset) and an
output folder:

```bash
# Linux or WSL
./run.sh /path/to/scene /path/to/output
```

```bat
REM Windows
run.bat C:\path\to\scene C:\path\to\output
```

The outputs go to `output/<scene>/`: the material `.ply`, preview renders (in `test/`), and the
training checkpoints.

## Importing into Unreal Engine

The pipeline exports a material PLY. This is not a native Unreal format, so to bring it in you
use the importer plugin developed by the University of Abertay as part of CoSTAR. The plugin
reads the PLY and its material channels (albedo, roughness, metallic) into Unreal.

## How it works

```
[0] preprocess   raw photos or video, COLMAP camera calibration (first run only)
[1] Stage 1      geometry and appearance (SH colour)
[2] bake         occlusion volumes (auto sized to the scene)
[3] Stage 2      material and lighting decomposition (PBR)
[4] render       preview PBR, normals and material channels
[5] export       material PLY
```

The pipeline and its options live in [packaging/](packaging/) (`run_pipeline.sh`,
`preprocess.sh`, the launchers, and the `Dockerfile`).

## Building the image yourself

```bash
docker build -f packaging/Dockerfile -t ghcr.io/costar/gs-ir:latest .
```

Run this from the repository root. The build fetches a pinned nvdiffrast and compiles the CUDA
extensions for the full GPU range.

## Repository layout

| Path | What it is |
|------|------------|
| `packaging/` | Docker image, one command pipeline, launchers, user and tester guides |
| `scene/nerfosr_reader.py` | NeRF-OSR dataparser (our addition) |
| `extract_ply.py` | material PLY exporter (our addition) |
| `train.py`, `baking.py`, `render.py`, `relight.py` | the GS-IR method (upstream) |
| `README_GSIR.md` | the original upstream GS-IR README (method, datasets, research commands) |

## Licensing

Use is non-commercial. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## Credits and origins

This is a fork of GS-IR by Zhihao Liang et al. See [README_GSIR.md](README_GSIR.md) for
the original method, datasets and research usage, and please cite their paper if you use this in
research:

```txt
@article{liang2023gs,
  title={GS-IR: 3D Gaussian Splatting for Inverse Rendering},
  author={Liang, Zhihao and Zhang, Qi and Feng, Ying and Shan, Ying and Jia, Kui},
  journal={arXiv preprint arXiv:2311.16473},
  year={2023}
}
```

GS-IR builds on [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
(Inria and MPII) and [nvdiffrast](https://github.com/NVlabs/nvdiffrast) (NVIDIA). Contributors
to this fork are listed in [AUTHORS.md](AUTHORS.md).
