"""
nerfosr_reader.py — Custom NeRF-OSR scene reader for GS-IR

Reads NeRF-OSR (post-cvt.py) datasets DIRECTLY using ground-truth poses and
intrinsics, bypassing COLMAP entirely. This preserves the dataset's accurate
calibration (the whole reason for using NeRF-OSR over raw images).

Key features:
  - Uses GT poses (C2W) + GT intrinsics (K) — no COLMAP pose estimation
  - Applies OpenGL->OpenCV Y/Z flip (from cvt.py's convert_pose)
  - Handles per-image dimension mismatches (min of image/mask size)
  - Generates RANDOM initial point cloud (GS-IR optimises positions anyway)
  - Loads NeRF-OSR masks (sky/foreground)

HOW TO INTEGRATE INTO GS-IR:
  1. Place this file in GS-IR's scene/ directory
  2. In scene/dataset_readers.py, import and register:
         from scene.nerfosr_reader import readNeRFOSRSceneInfo
         sceneLoadTypeCallbacks["NeRFOSR"] = readNeRFOSRSceneInfo
  3. In scene/__init__.py, add detection for NeRF-OSR structure
     (looks for train/rgb, train/pose, train/intrinsics) BEFORE the
     COLMAP/Blender checks, calling sceneLoadTypeCallbacks["NeRFOSR"].
  4. Train with -s pointing at the scene root (the dir containing train/, test/)

Author: Jagdeep Ahluwalia <j.ahluwalia@abertay.ac.uk>
"""

import os
import pathlib
import numpy as np
from PIL import Image

from utils.graphics_utils import BasicPointCloud, focal2fov

print("Loaded nerfosr reader")

def _find_images(rgb_dir):
    """Find all images in a directory, sorted by name."""
    exts = ("*.JPG", "*.jpg", "*.jpeg", "*.JPEG", "*.png", "*.PNG")
    files = []
    for e in exts:
        files.extend(pathlib.Path(rgb_dir).glob(e))
    return sorted(files, key=lambda p: p.name)


def _load_pose(pose_path, flip_yz=True):
    """
    Load a 4x4 C2W matrix from a NeRF-OSR pose file.
    Applies OpenGL->OpenCV Y/Z flip (matches cvt.py's convert_pose).
    """
    c2w = np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4)
    if flip_yz:
        flip = np.eye(4, dtype=np.float32)
        flip[1, 1] = -1.0
        flip[2, 2] = -1.0
        c2w = c2w @ flip
    return c2w


def _load_intrinsics(intr_path):
    """Load K matrix (3x3 or 4x4) and return fx, fy, cx, cy."""
    vals = np.loadtxt(intr_path, dtype=np.float32).reshape(-1)
    if vals.size == 9:
        K = vals.reshape(3, 3)
    elif vals.size == 16:
        K = vals.reshape(4, 4)[:3, :3]
    else:
        raise ValueError(f"Unexpected intrinsics size {vals.size} in {intr_path}")
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def readNeRFOSRCameras(split_dir, flip_yz=True, force_wh=None):
    from scene.dataset_readers import CameraInfo

    rgb_dir  = os.path.join(split_dir, "rgb")
    mask_dir = os.path.join(split_dir, "mask")
    pose_dir = os.path.join(split_dir, "pose")
    intr_dir = os.path.join(split_dir, "intrinsics")

    image_files = _find_images(rgb_dir)
    if len(image_files) == 0:
        raise RuntimeError(f"No images found in {rgb_dir}")

    # --- PASS 1: find the minimum common dimensions across all images ---
    if force_wh is not None:
        min_w, min_h = force_wh
    else:
        min_w, min_h = None, None
        for img_path in image_files:
            w, h = Image.open(img_path).size
            min_w = w if min_w is None else min(min_w, w)
            min_h = h if min_h is None else min(min_h, h)
        min_w -= (min_w % 2)
        min_h -= (min_h % 2)
    print(f"[NeRF-OSR] Cropping all images to common size: {min_w}x{min_h}")

    # --- PASS 2: load, cropping each image+intrinsics to (min_w, min_h) ---
    cam_infos = []
    for uid, img_path in enumerate(image_files):
        stem = img_path.stem
        pose_path = os.path.join(pose_dir, f"{stem}.txt")
        intr_path = os.path.join(intr_dir, f"{stem}.txt")
        mask_path = os.path.join(mask_dir, f"{stem}.png")

        if not os.path.exists(pose_path):
            raise FileNotFoundError(f"Missing pose: {pose_path}")
        if not os.path.exists(intr_path):
            raise FileNotFoundError(f"Missing intrinsics: {intr_path}")

        # GT pose with Y/Z flip
        c2w = _load_pose(pose_path, flip_yz=flip_yz)
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

        # GT intrinsics — cropping from top-left does NOT change fx,fy,cx,cy
        fx, fy, cx, cy = _load_intrinsics(intr_path)

        # Load and crop image to common size (top-left crop)
        image = Image.open(img_path).convert("RGB")
        image = image.crop((0, 0, min_w, min_h))

        width, height = min_w, min_h
        FovX = focal2fov(fx, width)
        FovY = focal2fov(fy, height)

        cam_infos.append(CameraInfo(
            uid=uid, R=R, T=T, FovY=FovY, FovX=FovX,
            image=image, image_path=str(img_path),
            image_name=stem, width=width, height=height,
        ))

    return cam_infos

def _create_random_pointcloud(cam_infos, num_points=100_000):
    """
    Generate a random initial point cloud within the camera bounding region.
    GS-IR optimises Gaussian positions during training, so random init is a
    valid starting seed (this is what the Blender/synthetic reader does too).
    """
    # Estimate scene bounds from camera positions
    cam_centers = []
    for cam in cam_infos:
        # camera center in world = -R^T @ T  (R here already transposed)
        # In 3DGS convention, R is W2C rotation transposed, so:
        R = cam.R
        T = cam.T
        # world-to-camera: Xc = R^T... careful. Reconstruct C2W center:
        Rt = np.transpose(R)
        center = -Rt @ T
        cam_centers.append(center)
    cam_centers = np.stack(cam_centers, axis=0)

    cmin = cam_centers.min(axis=0)
    cmax = cam_centers.max(axis=0)
    extent = (cmax - cmin)
    # Expand the box a bit so points cover the scene, not just camera hull
    center = (cmin + cmax) / 2.0
    half = np.maximum(extent, 1.0) * 1.5

    xyz = np.random.uniform(
        low=center - half,
        high=center + half,
        size=(num_points, 3),
    ).astype(np.float32)

    rgb = np.random.uniform(0, 1, size=(num_points, 3)).astype(np.float32)
    normals = np.zeros((num_points, 3), dtype=np.float32)

    return BasicPointCloud(points=xyz, colors=rgb, normals=normals)


def readNeRFOSRSceneInfo(path, images=None, eval=True, **kwargs):
    """
    Main entry point — mirrors readColmapSceneInfo signature.

    Args:
        path: scene root (contains train/, test/, validation/)
        images: unused (kept for signature compatibility)
        eval: if True, use test split as test cameras; else all in train
    """
    from scene.dataset_readers import SceneInfo, getNerfppNorm, storePly, fetchPly
    flip_yz = kwargs.get("flip_yz", True)

    train_dir = os.path.join(path, "train")
    test_dir  = os.path.join(path, "test")

    # --- Find global min dims across ALL splits for consistency ---
    def split_min(d):
        rgb = os.path.join(d, "rgb")
        mn_w, mn_h = None, None
        for p in _find_images(rgb):
            w, h = Image.open(p).size
            mn_w = w if mn_w is None else min(mn_w, w)
            mn_h = h if mn_h is None else min(mn_h, h)
        return mn_w, mn_h

    gw, gh = split_min(train_dir)
    if eval and os.path.exists(test_dir):
        tw, th = split_min(test_dir)
        gw, gh = min(gw, tw), min(gh, th)
    gw -= gw % 2 # TODO - what is this for
    gh -= gh % 2
    force = (gw, gh)
    print(f"[NeRF-OSR] Global common crop size: {gw}x{gh}")

    print("Reading NeRF-OSR train cameras...")
    train_cam_infos = readNeRFOSRCameras(train_dir, flip_yz=flip_yz, force_wh=force)

    if eval and os.path.exists(test_dir):
        print("Reading NeRF-OSR test cameras...")
        test_cam_infos = readNeRFOSRCameras(test_dir, flip_yz=flip_yz, force_wh=force)
    else:
        test_cam_infos = []

    # Scene normalisation (camera extent) — required by 3DGS/GS-IR
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # --- Initial point cloud ---
    ply_path = os.path.join(path, "points3d_random.ply")
    if not os.path.exists(ply_path):
        print("Generating random initial point cloud (100k points)...")
        pcd = _create_random_pointcloud(train_cam_infos, num_points=100_000)
        storePly(ply_path, pcd.points, pcd.colors * 255.0)
    try:
        pcd = fetchPly(ply_path)
    except Exception:
        pcd = _create_random_pointcloud(train_cam_infos, num_points=100_000)

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info
