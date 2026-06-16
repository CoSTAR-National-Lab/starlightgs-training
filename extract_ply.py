# PLY material exporter for GS-IR checkpoints.
# Original author: Jamie Buttenshaw <jamie.buttenshaw@gmail.com>

import os
import torch
import numpy as np
import argparse
from pathlib import Path

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def normalize(v, eps=1e-8):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)

def write_binary_ply(path, fields):
    names = list(fields.keys())
    N = len(next(iter(fields.values())))

    for k, v in fields.items():
        assert len(v) == N, f"Field {k} has wrong length"

    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {N}",
    ]
    for name in names:
        header.append(f"property float {name}")
    header.append("end_header\n")

    with open(path, "wb") as f:
        f.write("\n".join(header).encode("ascii"))
        stacked = np.column_stack([fields[n].astype(np.float32) for n in names])
        f.write(stacked.astype("<f4").tobytes())

def parse_arguments():
    """Parses and returns the command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract ply from checkpoint")
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        help="Path to the checkpoint file containing the gaussian info."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="out.ply",
        help="Name of the output ply file."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs",
        help="Path where the output ply file will be saved."
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    #pkl_path = "~/workspace/GS-IR/outputs/stjohann_gt/chkpnt35000.pth"
    #out_ply  = "~/workspace/GS-IR/outputs/stjohann_gt/out.ply"

    pkl_path = Path(args.checkpoint_file)
    out_ply  = Path(args.output_path) / args.output_file

    print(f"Loading ckeckpoint: {pkl_path}")
    data = torch.load(pkl_path, map_location="cpu", weights_only=False)

    # Hard-coded by looking at implemtation of GS-IR (see https://github.com/lzhnb/GS-IR/blob/main/scene/gaussian_model.py)
    # data["gaussians"] contains all the data that we need to export in PLY
    # This dict contains all the information needed to know how to interpret the data in data["gaussians"]
    # Each parameter needs to know the index in data["gaussians"] and the activation function to apply (if any)
    parameters = {
        "xyz": (1, None),
        "f_dc": (2, None),
        "f_rest": (3, None),
        "scaling": (4, None), # exp is applied on splat import
        "rotation": (5, torch.nn.functional.normalize),
        "opacity": (6, torch.sigmoid),
        "normal": (7, torch.nn.functional.normalize),
        "albedo": (8, torch.sigmoid),
        "roughness": (9, torch.sigmoid),
        "metallic": (10, torch.sigmoid),
    }

    splat_data = {}

    for param, (idx, activation) in parameters.items():

        param_data = data["gaussians"][idx]

        if activation is not None:
            param_data = activation(param_data)

        splat_data[param] = to_numpy(param_data).astype(np.float32)

    N = splat_data["xyz"].shape[0]

    means     = splat_data["xyz"].reshape(N, 3)
    normals   = splat_data["normal"].reshape(N, 3)
    albedo    = splat_data["albedo"].reshape(N, 3)
    opacities = splat_data["opacity"].reshape(N, 1)
    scales    = splat_data["scaling"].reshape(N, 3)
    quats     = splat_data["rotation"].reshape(N, 4)
    metallic  = splat_data["metallic"].reshape(N, 1)
    roughness = splat_data["roughness"].reshape(N, 1)

    fields = {
        "x": means[:, 0],
        "y": means[:, 1],
        "z": means[:, 2],
        "nx": normals[:, 0],
        "ny": normals[:, 1],
        "nz": normals[:, 2],
        "f_dc_0": albedo[:, 0], # Actually export albedo to match what UE exporter expects (it doesn't handle both original lit base colour and albedo, so we only export albedo)
        "f_dc_1": albedo[:, 1],
        "f_dc_2": albedo[:, 2],
        "metallicFactor": metallic,
        "roughnessFactor": roughness,
        "opacity": opacities,
        "scale_0": scales[:, 0],
        "scale_1": scales[:, 1],
        "scale_2": scales[:, 2],
        "rot_0": quats[:, 0],
        "rot_1": quats[:, 1],
        "rot_2": quats[:, 2],
        "rot_3": quats[:, 3]
    }

    os.makedirs(os.path.dirname(out_ply), exist_ok=True)
    write_binary_ply(out_ply, fields)
    print(f"Wrote PLY: {out_ply}")

    print("\nSanity check:")
    print("means     ", means.shape, means.dtype, means.min(), means.max())
    print("normals   ", normals.shape, normals.dtype, normals.min(), normals.max())
    print("albedo    ", albedo.shape, albedo.dtype, albedo.min(), albedo.max())
    print("metallic  ", metallic.shape, metallic.dtype, metallic.min(), metallic.max())
    print("roughness ", roughness.shape, roughness.dtype, roughness.min(), roughness.max())
    print("opacity   ", opacities.shape, opacities.dtype, opacities.min(), opacities.max())
    print("scales    ", scales.shape, scales.dtype, scales.min(), scales.max())
    print("quats     ", quats.shape, quats.dtype, quats.min(), quats.max())

