# GS-IR Quick Start

Turn a set of photos of a scene into a **material PLY** (albedo / roughness / metallic +
geometry) you can drop into Unreal Engine, plus preview renders, with a single command.
No Python, no coding.

> ⚠️ **Licensing, non-commercial use only.** This tool bundles the Inria 3D Gaussian
> Splatting rasterizer, which is licensed for **research / evaluation purposes only**. Do
> not use it, or anything it produces, for commercial work without the appropriate
> licence. See [`THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md).

---

## 1. What you need

**Hardware**

- An **NVIDIA GPU** from 2018 or newer (RTX 20-, 30-, 40- or 50-series, or a workstation/datacentre
  card like A100 / H100 / H200). **≥ 12 GB VRAM recommended.**
  - 8 GB cards work, but you must lower the resolution, see *Out of memory* below.
- **NVIDIA driver 570 or newer** (needed for CUDA 12.8; RTX 50-series cards already have this).
  If yours is older, update it from [nvidia.com/drivers](https://www.nvidia.com/en-us/drivers/)
  (on Windows the NVIDIA App does it in a click; a driver update is reversible).

**Not supported:** Apple Silicon / Intel Macs (no NVIDIA GPU = no CUDA), AMD GPUs.
On those machines this tool cannot run at all.

**Software**

- **Windows 10/11:** install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
  and enable the WSL2 + GPU integration (Docker Desktop → Settings → Resources → WSL, and
  make sure the NVIDIA driver is installed on Windows).
- **Linux:** install Docker + the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (this is what lets containers see the GPU).

Check the GPU is visible to Docker:

```bash
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.03-py3 nvidia-smi
```

If that prints your GPU, you're ready.

---

## 2. Get the image

```bash
docker pull ghcr.io/costar/gs-ir:latest
```

---

## 3. Prepare your scene folder

The tool auto-detects what you give it. Your scene folder can be **either**:

**A. Just your raw capture** (the easy case, the tool calibrates it for you):

| You have…                         | What happens |
|-----------------------------------|--------------|
| a **video** file (`.mp4`/`.mov`)  | frames are extracted, then COLMAP recovers the camera poses |
| **photos** loose in the folder, or in an `input/` subfolder | COLMAP recovers the camera poses |

> Put photos **loose in the scene folder** or inside a subfolder called **`input/`**.

The first run does this **once** and saves the result back into your scene folder (it creates
`input/` with your raw frames, plus `images/` and `sparse/`). Later runs reuse it and skip the
slow calibration. **COLMAP calibration adds ~10–60 min** on the first run and needs photos with
**good overlap** (each area seen from several angles).

**B. An already-calibrated scene** (skips calibration):

| Format   | It has…                    |
|----------|----------------------------|
| COLMAP   | a `sparse/` folder         |
| NeRF-OSR | a `train/pose/` folder     |
| Blender  | a `transforms_train.json`  |

> Tip: for best results from photos, capture lots of overlapping views (think 100–300) all
> the way around the subject. Blurry or sparse captures may fail to calibrate. The easiest way
> is to capture a video around an object slowly at different highs and let the processing
> script do its magic.

---

## 4. Run it

Use the launcher for your OS. Give it your **scene folder** and an **output folder**:

**Linux / macOS-shell / WSL:**
```bash
./run.sh /path/to/my_scene /path/to/output
```

**Windows (Command Prompt):**
```bat
run.bat C:\path\to\my_scene C:\path\to\output
```

That's it. It will, in order: train the geometry, bake lighting occlusion, decompose the
materials, render previews, and export the PLY. **This takes roughly 1–3 hours** depending
on your GPU and scene size (the baking step alone is ~30–40 min).

If you'd rather not use the launcher script, the raw command is:
```bash
docker run --rm --gpus all \
  -e GSIR_SCENE=my_scene \
  -v /path/to/my_scene:/data -v /path/to/output:/output \
  ghcr.io/costar/gs-ir:latest /data
```
(`GSIR_SCENE` just names the output subfolder; the `run.sh`/`run.bat` launchers set it
for you from your folder name.)

---

## 5. Where are my results?

In the output folder you gave it, under a subfolder named after your scene:

```
output/<scene>/
  <scene>.ply          <- the material PLY for Unreal
  test/                <- preview renders: pbr/, normal/, renders/, gt/
  chkpnt30000.pth      <- checkpoints (keep these to re-export later without retraining)
  chkpnt35000.pth
```

---

## Troubleshooting

**"could not select device driver … capabilities: [[gpu]]"**: Docker can't see the GPU.
Re-check the `nvidia-smi` test in step 1 (driver installed? NVIDIA Container Toolkit installed
on Linux? WSL GPU integration enabled on Windows?).

**Out of memory / "CUDA out of memory"**: the tool trains at full resolution by default and
loads every image onto the GPU, so a large photo set can exhaust it. It warns you just before
training starts if it expects this. Two ways out, best quality first:

**1. Keep full quality, use system memory instead** (slower, but the images stay full size):
```bash
# Linux/WSL:
GSIR_DATA_DEVICE=cpu ./run.sh /path/to/my_scene /path/to/output
# Windows:
set GSIR_DATA_DEVICE=cpu
run.bat C:\path\to\my_scene C:\path\to\output
```

**2. Or lower the resolution** (faster, but less detail):
```bash
# Linux/WSL:
GSIR_RESOLUTION=2 ./run.sh /path/to/my_scene /path/to/output
# Windows:
set GSIR_RESOLUTION=2
run.bat C:\path\to\my_scene C:\path\to\output
```
Try `GSIR_RESOLUTION=4` if `2` still runs out. **Note:** NeRF-OSR scenes must stay at
`GSIR_RESOLUTION=1` (their calibration requires it).

**"no images or video found"**: your folder has none of the inputs in step 3 (a video, loose
photos, an `input/` folder, or an already-calibrated scene).

**"COLMAP could not reconstruct this scene"**: calibration failed: usually too few photos or
too little overlap. Capture more views (100–300), moving slowly so consecutive frames share
lots of content. A slow orbit video is the easiest way. For video, raise `GSIR_TARGET_FRAMES`.

**COLMAP step says "Killed"**: it ran out of RAM during calibration (common with very
high-resolution photos on a modest machine). Lower the feature-detection size, e.g.
`GSIR_COLMAP_MAX_IMAGE_SIZE=1600` (this does not lower your training image quality).

**It's very slow / seems stuck**: calibration (`[0/5]`, first run only) and training genuinely
take a while; the progress banners (`[0/5] … [5/5]`) tell you which stage you're in. As long as
one is printed you're fine.

**Advanced knobs** (optional, via environment variables):
- Pipeline: `GSIR_RESOLUTION` (1 = full resolution), `GSIR_ITERS_S1` (30000),
  `GSIR_ITERS_S2` (35000), `GSIR_OCCLU_RES` (256), `GSIR_OCCLUSION` (0.4),
  `GSIR_SCENE` (output subfolder name).
- `GSIR_DATA_DEVICE` (cuda). Set to `cpu` to hold the training images in system memory
  rather than on the GPU. This keeps full image quality and greatly reduces GPU memory
  use, at the cost of slower training, so it is the best option if you run out of memory
  but want maximum quality. Needs enough system RAM to hold all the images.
- Calibration (raw photos/video): `GSIR_TARGET_FRAMES` (300, frames extracted from a video),
  `GSIR_MATCHER` (auto | exhaustive | sequential), `GSIR_CAMERA` (OPENCV),
  `GSIR_COLMAP_MAX_IMAGE_SIZE` (2000), `GSIR_COLMAP_GPU` (0).

---

## Advanced (optional)

*You don't need this for normal use.* The container normally runs the whole pipeline, but
power users can override that to run just one script. Because the checkpoints are kept in
your output folder, you can re-do the final step without retraining.

**Re-export the PLY from an existing run** (no retraining, it takes seconds, not hours):
```bash
docker run --rm --gpus all -v /path/to/output:/output \
  --entrypoint python ghcr.io/costar/gs-ir:latest \
  extract_ply.py --checkpoint-file /output/<scene>/chkpnt35000.pth \
                 --output-path /output/<scene> --output-file <scene>.ply
```

**Open a shell inside the container** (to inspect files or run commands manually):
```bash
docker run --rm -it --entrypoint bash ghcr.io/costar/gs-ir:latest
```

`--entrypoint <program>` replaces the automatic pipeline for that one run; everything after
the image name becomes arguments to that program.
