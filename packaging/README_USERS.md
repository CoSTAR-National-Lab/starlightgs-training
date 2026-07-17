# GS-IR — Quick Start

Turn a set of photos of a scene into a **material PLY** (albedo / roughness / metallic +
geometry) you can drop into Unreal Engine, plus preview renders, with a single command.
No Python, no coding.

---

## 1. What you need

**Hardware**

- An **NVIDIA GPU** from 2018 or newer (RTX 20-, 30-, 40-series, or a workstation/datacentre
  card like A100 / H100 / H200). **≥ 12 GB VRAM recommended.**
  - 8 GB cards work, but you must lower the resolution — see *Out of memory* below.
- NVIDIA driver **545 or newer**.

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
docker run --rm --gpus all nvcr.io/nvidia/pytorch:24.04-py3 nvidia-smi
```

If that prints your GPU, you're ready.

---

## 2. Get the image

```bash
docker pull ghcr.io/costar/gs-ir:latest
```

---

## 3. Prepare your scene folder

The tool auto-detects three formats. Your scene folder must be **one** of:

| Format   | It has…                    |
|----------|----------------------------|
| NeRF-OSR | a `train/pose/` folder     |
| COLMAP   | a `sparse/` folder         |
| Blender  | a `transforms_train.json`  |

If you only have raw photos, you first need to run camera calibration (COLMAP) to produce a
`sparse/` folder. You can follow the process explained [here](https://github.com/fyusion/llff#general-step-by-step-usage).
TODO - Process the raw datasets within this image.

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

**"could not select device driver … capabilities: [[gpu]]"** — Docker can't see the GPU.
Re-check the `nvidia-smi` test in step 1 (driver installed? NVIDIA Container Toolkit installed
on Linux? WSL GPU integration enabled on Windows?).

**Out of memory / "CUDA out of memory"** — your GPU has too little VRAM for full resolution.
Run at half resolution by adding an environment variable:
```bash
# Linux/WSL:
GSIR_RESOLUTION=2 ./run.sh /path/to/my_scene /path/to/output
# Windows:
set GSIR_RESOLUTION=2
run.bat C:\path\to\my_scene C:\path\to\output
```
Try `GSIR_RESOLUTION=4` if `2` still runs out. **Note:** NeRF-OSR scenes must stay at
`GSIR_RESOLUTION=1` (their calibration requires it).

**"is not a recognised scene format"** — your folder isn't one of the three formats in step 3.

**It's very slow / seems stuck** — training genuinely takes hours; the progress banners
(`[1/5] … [5/5]`) tell you which stage you're in. As long as one is printed you're fine.

**Advanced knobs** (optional, via environment variables): `GSIR_ITERS_S1` (default 30000),
`GSIR_ITERS_S2` (35000), `GSIR_OCCLU_RES` (256), `GSIR_OCCLUSION` (0.4), `GSIR_SCENE`
(output subfolder name).

---

## Advanced (optional) — run a single step

*You don't need this for normal use.* The container normally runs the whole pipeline, but
power users can override that to run just one script. Because the checkpoints are kept in
your output folder, you can re-do the final step without retraining.

**Re-export the PLY from an existing run** (no retraining — seconds, not hours):
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
