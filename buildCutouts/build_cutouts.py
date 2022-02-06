# Unused, more general version of artwin build cutouts script.
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


def projectMesh(scene, k, R, t, debug):
    # In OpenGL, camera points toward -z by default, hence we don't need rFix like in the MATLAB code

    camera = pyrender.IntrinsicsCamera(
        k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    )

    camera_pose = np.eye(4)
    camera_pose[0:3, 0:3] = R.T
    camera_pose[0:3, 3] = t
    camera_pose[:, 1:3] *= -1
    cameraNode = scene.add(camera, pose=camera_pose)

    scene._ambient_light = np.ones((3,))
    sensorWidth = int(2 * k[0, 2])
    sensorHeight = int(2 * k[1, 2])
    r = pyrender.OffscreenRenderer(sensorWidth, sensorHeight, point_size=5)
    meshProjection, depth = r.render(scene) # TODO: this thing consumes ~14 GB RAM!!!
    r.delete()

    # XYZ cut
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

    scene.remove_node(cameraNode)

    return meshProjection, xyzCut, depth, XYZpc

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

def cutoutFromPhoto(scene, photo_path, output_root):
    stem = photo_path.stem.strip("_reference")

    params_in_path = photo_path.parent / (stem + "_params.json")

    mesh_out_path = output_root / "meshes" / ("mesh_" + stem + ".png")
    cutout_out_path = output_root / "cutouts" / ("cutout_" + stem + ".png")
    mat_out_path = output_root / "matfiles" / (cutout_out_path.name + ".mat")
    pose_out_path = output_root / "poses" / (cutout_out_path.name + ".mat")
    depth_out_path = output_root / "depthmaps" / ("depth_" + stem + ".png")

    # if mesh_out_path.exists():
    #     return

    with open(params_in_path, 'r') as f:
        params = json.load(f)

    calibration_mat = np.array(params["calibration_mat"])
    rotationMatrix = np.array(params["x_rot_mat"]) @ np.array(params["z_rot_mat"]) @ np.array(params["pano_rot_mat"]).T
    translation = np.array(params["pano_translation"])
    RGBcut, XYZcut, depth, _ = projectMesh(scene, calibration_mat, rotationMatrix, translation, False)

    sio.savemat(mat_out_path, {'RGBcut': cv2.imread(str(photo_path), cv2.IMREAD_UNCHANGED), 'XYZcut': XYZcut})
    sio.savemat(pose_out_path, {'R': rotationMatrix, 'position': translation})
    if not cutout_out_path.exists():
        os.link(photo_path, cutout_out_path)

    # Debug
    plt.imsave(depth_out_path, depth, cmap=plt.cm.gray_r)
    cv2.imwrite(
        str(depth_out_path.parent / (depth_out_path.stem + "_uint16.png")),
        depth.astype(np.uint16),
    )
    # For possible further recalculation, npz with raw depth map is also saved.
    np.save(
        str(depth_out_path) + ".npy",
        depth,
    )
    plt.imsave(mesh_out_path, RGBcut)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_root", type=Path, help="Root input data folder", default='/nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/2019-09-28_08.31.29/images/'
    )
    parser.add_argument(
        "--input_ply_path",
        type=str,
        help="path to point cloud or mesh .ply file",
        default='/nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/2019-09-28_08.31.29/2019-09-28_08.31.29.ply'
    )
    parser.add_argument(
        "--output_root", type=Path, help="path to write output data", default='/nfs/projects/artwin/experiments/artwin-inloc/2019-09-28_08.31.29'
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

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0])
    mesh = load_ply(args.input_ply_path)
    scene.add(mesh)

    model = args.output_root / "model" / "model_rotated.obj"
    if not model.exists():
        os.link(args.input_ply_path, model)

    photos = list(args.input_root.glob("*.png"))
    for idx, photo_path in enumerate(photos):
        print(f"Processing {idx + 1}/{len(photos)}")
        cutoutFromPhoto(scene, photo_path, args.output_root)