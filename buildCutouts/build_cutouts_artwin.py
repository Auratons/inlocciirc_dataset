import argparse
import cv2
import json
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import sys
import open3d as o3d
from pathlib import Path

pm = Path(__file__).parent / '..' / '..' / 'functions' / 'inLocCIIRC_utils' / 'projectMesh'
sys.path.insert(0, str(pm.resolve()))
from projectMesh import buildXYZcut

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
os.environ['NVIDIA_DRIVER_CAPABILITIES'] = 'compute,graphics,utility,video'
os.environ["PYOPENGL_PLATFORM"] = "egl"
# When ran with SLURM on a multigpu node, scheduled on other than GPU0, we need
# to set this or we get an egl initialization error.
os.environ["EGL_DEVICE_ID"] = os.environ.get("SLURM_JOB_GPUS", "0").split(",")[0]

import pyrender


def project_mesh_core(depth, k, R, t, debug):
    sensorWidth = int(2 * k[0, 2])
    sensorHeight = int(2 * k[1, 2])
    f = (k[0, 0] + k[1, 1]) / 2
    scaling = 1.0 / f
    spaceCoordinateSystem = np.eye(3)
    sensorCoordinateSystem = np.matmul(R, spaceCoordinateSystem)
    sensorXAxis = sensorCoordinateSystem[:, 0]
    sensorYAxis = -sensorCoordinateSystem[:, 1]
    # make camera point toward -z by default, as in OpenGL
    cameraDirection = -sensorCoordinateSystem[:, 2] # unit vector

    xyzCut, pts = buildXYZcut(
        sensorWidth, sensorHeight,
        t, cameraDirection, scaling,
        sensorXAxis, sensorYAxis, depth
    )

    XYZpc = -1
    if debug:
        XYZpc = o3d.geometry.PointCloud()
        XYZpc.points = o3d.utility.Vector3dVector(pts)

    return xyzCut, XYZpc

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

def load_ply(ply_path, voxel_size=None):
    # Loading the mesh / pointcloud
    m = trimesh.load(ply_path)
    if isinstance(m, trimesh.PointCloud):
        if voxel_size is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.asarray(m.vertices))
            pcd.colors = o3d.utility.Vector3dVector(
                np.asarray(m.colors, dtype=np.float64)[:, :3] / 255
            )
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            mesh = o3d_to_pyrenderer(pcd)
        else:
            points = m.vertices.copy()
            colors = m.colors.copy()
            mesh = pyrender.Mesh.from_points(points, colors)
    elif isinstance(m, trimesh.Trimesh):
        if voxel_size is not None:
            m2 = m.as_open3d
            m2.vertex_colors = o3d.utility.Vector3dVector(
                np.asarray(m.visual.vertex_colors, dtype=np.float64)[:, :3] / 255
            )
            m2 = m2.simplify_vertex_clustering(
                voxel_size=voxel_size,
                contraction=o3d.geometry.SimplificationContraction.Average,
            )
            mesh = o3d_to_pyrenderer(m2)
        else:
            mesh = pyrender.Mesh.from_trimesh(m)
    else:
        raise NotImplementedError(
            "Unsupported 3D object. Supported format is a `.ply` pointcloud or mesh."
        )
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

    with open(params_in_path, 'r') as f:
        params = json.load(f)

    depth = np.load(str(depth_npy))
    calibration_mat = np.array(params["calibration_mat"])
    rotationMatrix = np.array(params["x_rot_mat"]) @ np.array(params["z_rot_mat"]) @ np.array(params["pano_rot_mat"]).T
    translation = np.array(params["pano_translation"])
    XYZcut, _ = project_mesh_core(depth, calibration_mat, rotationMatrix, translation, False)

    sio.savemat(mat_out_path, {'RGBcut': cv2.imread(str(mesh_projection), cv2.IMREAD_UNCHANGED), 'XYZcut': XYZcut})
    sio.savemat(pose_out_path, {'R': rotationMatrix, 'position': translation, 'calibration_mat': calibration_mat})
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_root", type=Path, help="Root input data folder", default='/nfs/projects/artwin/experiments/hololens_mapper/joined_dataset/train'
    )

    parser.add_argument(
        "--input_mapping", type=Path, help="Mapping from n-tuple to source img", default='/nfs/projects/artwin/experiments/hololens_mapper/joined_dataset/mapping.txt'
    )

    parser.add_argument(
        "--output_root", type=Path, help="path to write output data", default='/nfs/projects/artwin/experiments/artwin-inloc/joined_dataset_train'
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

    def reverse_dict(d):
        return {v: k for k, v in d.items()}

    # Get mapping from reference images in joined dataset for nriw training to
    # reference images in a concrete nriw training dataset from which the joined
    # one was generated.
    with open(args.input_mapping, 'r') as f:
        sub_mapping_1 = json.load(f)
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
        with open(mp, 'r') as f:
            lines = f.readlines()
            # Filter lines only for used source (train/val/test)
            lines = filter(lambda line: str(args.input_root.name).upper() in line, lines)
            lines = [str.join(" ", line.split(" ")[:-1]) for line in lines]  # Get rid of trailing TRAIN/DEV/TEST
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
