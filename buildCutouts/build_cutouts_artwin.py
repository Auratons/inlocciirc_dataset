# pylint: disable=wrong-import-position,missing-module-docstring,missing-function-docstring
import os
import sys
from pathlib import Path

sys.path.insert(
    0,
    str((Path(__file__).parent / "../../functions/inLocCIIRC_utils/projectMesh").resolve())
)

os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "compute,graphics,utility,video"
os.environ["PYOPENGL_PLATFORM"] = "egl"
# When ran with SLURM on a multigpu node, scheduled on other than GPU0, we need
# to set this or we get an egl initialization error.
os.environ["EGL_DEVICE_ID"] = os.environ.get("SLURM_JOB_GPUS", "0").split(",")[0]

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

import cv2
import open3d as o3d
import pyrender
import trimesh
from projectMesh import buildXYZcut


def project_mesh_core(depth, k, R, t, debug):
    sensor_width = int(2 * k[0, 2])
    sensor_height = int(2 * k[1, 2])
    focal_length = (k[0, 0] + k[1, 1]) / 2
    scaling = 1.0 / focal_length
    space_coord_system = np.eye(3)
    sensor_coord_system = np.matmul(R, space_coord_system)
    sensor_x_axis = sensor_coord_system[:, 0]
    sensor_y_axis = -sensor_coord_system[:, 1]
    # make camera point toward -z by default, as in OpenGL
    camera_dir = -sensor_coord_system[:, 2] # unit vector

    xyz_cut, pts = buildXYZcut(
        sensor_width, sensor_height,
        t, camera_dir, scaling,
        sensor_x_axis, sensor_y_axis, depth
    )

    xyz_pc = -1
    if debug:
        xyz_pc = o3d.geometry.PointCloud()
        xyz_pc.points = o3d.utility.Vector3dVector(pts)

    return xyz_cut, xyz_pc

def o3d_to_pyrenderer(mesh_or_pt):
    if isinstance(mesh_or_pt, o3d.geometry.PointCloud):
        points = np.asarray(mesh_or_pt.points).copy()
        colors = np.asarray(mesh_or_pt.colors).copy()
        mesh = pyrender.Mesh.from_points(points, colors)
    elif isinstance(mesh_or_pt, o3d.geometry.TriangleMesh):
        mesh = trimesh.Trimesh(
            np.asarray(mesh_or_pt.vertices),
            np.asarray(mesh_or_pt.triangles),
            vertex_colors=np.asarray(mesh_or_pt.vertex_colors),
        )
        mesh = pyrender.Mesh.from_trimesh(mesh)
    else:
        raise NotImplementedError()
    return mesh

def cutoutFromPhoto(mapping, photo_path, output_root):
    stem = photo_path.stem.strip("_reference")

    source_photo = Path(mapping[str(photo_path)])
    params_in_path = source_photo.parent / (source_photo.stem.strip("_reference") + "_params.json")

    depth_npy = photo_path.parent / (stem + "_depth.npy")
    mesh_projection = photo_path.parent / (stem + "_color.png")
    cutout_reference = photo_path

    mesh_out_path = output_root / "meshes" / ("mesh_" + stem + ".png")
    cutout_out_path = output_root / "cutouts" / ("cutout_" + stem + ".png")
    mat_out_path = output_root / "matfiles" / (cutout_out_path.name + ".mat")
    pose_out_path = output_root / "poses" / (cutout_out_path.name + ".mat")
    depth_out_path = output_root / "depthmaps" / ("depth_" + stem + ".png")

    if mesh_out_path.exists():
        return

    with open(params_in_path, "r") as file:
        params = json.load(file)

    depth = np.load(str(depth_npy))
    calibration_mat = np.array(params["calibration_mat"])
    rotation_mat = np.array(params["x_rot_mat"]) @ np.array(params["z_rot_mat"]) @ np.array(params["pano_rot_mat"]).T
    translation = np.array(params["pano_translation"])
    XYZcut, _ = project_mesh_core(depth, calibration_mat, rotation_mat, translation, False)

    sio.savemat(
        mat_out_path,
        {"RGBcut": cv2.imread(str(mesh_projection), cv2.IMREAD_UNCHANGED), "XYZcut": XYZcut}
    )
    sio.savemat(
        pose_out_path,
        {"R": rotation_mat, "position": translation, "calibration_mat": calibration_mat}
    )
    if not cutout_out_path.exists():
        os.link(cutout_reference, cutout_out_path)

    # Debug
    plt.imsave(depth_out_path, depth, cmap=plt.cm.gray_r)
    cv2.imwrite(
        str(depth_out_path.parent / (depth_out_path.stem + "_uint16.png")),
        depth.astype(np.uint16),
    )
    # For possible further recalculation, npz with raw depth map is also saved.
    if not Path(str(depth_out_path) + ".npy").exists():
        os.link(depth_npy, str(depth_out_path) + ".npy")
    if not mesh_out_path.exists():
        os.link(mesh_projection, mesh_out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_root",
        type=Path,
        help="Root input data folder",
        default="/nfs/projects/artwin/experiments/hololens_mapper/joined_dataset/train"
    )
    parser.add_argument(
        "--input_mapping",
        type=Path,
        help="Mapping from n-tuple to source img",
        default="/nfs/projects/artwin/experiments/hololens_mapper/joined_dataset/mapping.txt"
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        help="path to write output data",
        default="/nfs/projects/artwin/experiments/artwin-inloc/joined_dataset_train"
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "model").mkdir(exist_ok=True)
    (args.output_root / "sweepData").mkdir(exist_ok=True)
    (args.output_root / "depthmaps").mkdir(exist_ok=True)
    (args.output_root / "meshes").mkdir(exist_ok=True)
    (args.output_root / "cutouts").mkdir(exist_ok=True)
    (args.output_root / "matfiles").mkdir(exist_ok=True)
    (args.output_root / "poses").mkdir(exist_ok=True)

    def reverse_dict(dictionary):
        return {v: k for k, v in dictionary.items()}

    # Get mapping from reference images in joined dataset for nriw training to
    # reference images in a concrete nriw training dataset from which the joined
    # one was generated.
    with open(args.input_mapping, "r") as file:
        sub_mapping_1 = json.load(file)
    sub_mapping_1 = reverse_dict(sub_mapping_1)

    # Get roots of nriw training datasets from which a joined dataset was generated.
    source_roots = {}
    for path in [Path(k).parent.parent for k in sub_mapping_1.values()]:
        source_roots[str(path)] = 1
    source_roots = list(source_roots.keys())

    # Get mapping from partial nriw training datasets to source references
    # generated by matlab from artwin panoramas
    sub_mappings_2 = []
    for mp in [Path(root) / "mapping.txt" for root in source_roots]:
        with open(mp, "r") as file:
            lines = file.readlines()
            # Filter lines only for used source (train/val/test)
            lines = filter(lambda line: str(args.input_root.name).upper() in line, lines)
            # Get rid of trailing TRAIN/DEV/TEST
            lines = [str.join(" ", line.split(" ")[:-1]) for line in lines]
            line_tuples = [tuple(line.split(" -> ")) for line in lines]  # (source, dest)
            sub_map = {}
            for wut in zip(line_tuples):
                v, k = wut[0]
                sub_map[str(Path(mp).parent / str(args.input_root.name) / f"{int(k):04n}_reference.png")] = f"/nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/{str.join('_', v.split('_')[0:2])}/images/{v}"
            sub_mappings_2.append(sub_map)

    # Get mapping from reference images in joined dataset for nriw training to
    # source references generated by matlab from artwin panoramas
    mapping = {}
    for k, v in sub_mapping_1.items():
        for sm in sub_mappings_2:
            if v in sm:
                mapping[k] = sm[v]

    photos = list(args.input_root.glob("*_reference.png"))
    for idx, photo_path in enumerate(photos):
        print(f"Processing {idx + 1}/{len(photos)}")
        cutoutFromPhoto(mapping, photo_path, args.output_root)
