# pylint: disable=wrong-import-position,missing-module-docstring,missing-function-docstring,no-member,c-extension-no-member
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
import distutils
import json
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

import cv2
import open3d as o3d
import pyrender
import read_model
import trimesh
from projectMesh import buildXYZcut


def projectMesh(mesh, k, r, t, debug):
    # In OpenGL, camera points toward -z by default, we don't need rFix like in the MATLAB code

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0])
    scene.add(mesh)

    camera = pyrender.IntrinsicsCamera(
        k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    )

    camera_pose = np.eye(4)
    camera_pose[0:3, 0:3] = r.T
    camera_pose[0:3, 3] = t
    camera_pose[:, 1:3] *= -1
    scene.add(camera, pose=camera_pose)

    sensor_width = int(2 * k[0, 2])
    sensor_height = int(2 * k[1, 2])
    renderer = pyrender.OffscreenRenderer(sensor_width, sensor_height, point_size=2)
    color, depth = renderer.render(scene, flags=pyrender.constants.RenderFlags.FLAT)
    renderer.delete()

    # XYZ cut
    focal_length = (k[0, 0] + k[1, 1]) / 2
    scaling = 1.0 / focal_length

    space_coord_system = np.eye(3)
    sensor_coord_system = np.matmul(r, space_coord_system)
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

    return color, xyz_cut, depth, xyz_pc

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

def get_colmap_file(colmap_path, file_stem):
    colmap_path = Path(colmap_path)
    fp = colmap_path / f"{file_stem}.bin"
    if not fp.exists():
        fp = colmap_path / f"{file_stem}.txt"
    return str(fp)

# Load camera matrices and names of corresponding src images from
# colmap images.bin and cameras.bin files from colmap sparse reconstruction
def load_cameras_colmap(images_fp, cameras_fp):
    if images_fp.endswith(".bin"):
        images = read_model.read_images_binary(images_fp)
    else:  # .txt
        images = read_model.read_images_text(images_fp)

    if cameras_fp.endswith(".bin"):
        cameras = read_model.read_cameras_binary(cameras_fp)
    else:  # .txt
        cameras = read_model.read_cameras_text(cameras_fp)

    src_img_nms = []
    K = []
    T = []
    R = []
    w = []
    h = []

    for i in images.keys():
        R.append(read_model.qvec2rotmat(images[i].qvec))
        T.append((images[i].tvec)[..., None])
        k = np.eye(3)
        camera = cameras[images[i].camera_id]
        if camera.model in ["SIMPLE_RADIAL", "SIMPLE_PINHOLE"]:
            k[0, 0] = cameras[images[i].camera_id].params[0]
            k[1, 1] = cameras[images[i].camera_id].params[0]
            k[0, 2] = cameras[images[i].camera_id].params[1]
            k[1, 2] = cameras[images[i].camera_id].params[2]
        elif camera.model in ["RADIAL", "PINHOLE"]:
            k[0, 0] = cameras[images[i].camera_id].params[0]
            k[1, 1] = cameras[images[i].camera_id].params[1]
            k[0, 2] = cameras[images[i].camera_id].params[2]
            k[1, 2] = cameras[images[i].camera_id].params[3]
        # TODO : Take other camera models into account + factorize
        else:
            raise NotImplementedError("Camera models not supported yet!")

        K.append(k)
        w.append(cameras[images[i].camera_id].width)
        h.append(cameras[images[i].camera_id].height)
        src_img_nms.append(images[i].name)

    return K, R, T, h, w, src_img_nms

def cutoutFromPhoto(index, mesh, photo_path, output_root, colmap_root):
    if colmap_root is None:
        stem = photo_path.stem.strip("_reference")
    else:
        stem = "{:04n}".format(index)

    mesh_out_path = output_root / "meshes" / ("mesh_" + stem + ".png")
    cutout_out_path = output_root / "cutouts" / ("cutout_" + stem + ".png")
    mat_out_path = output_root / "matfiles" / (cutout_out_path.name + ".mat")
    pose_out_path = output_root / "poses" / (cutout_out_path.name + ".mat")
    depth_out_path = output_root / "depthmaps" / ("depth_" + stem + ".png")

    # if mesh_out_path.exists():
    #     return

    if colmap_root is None:
        params_in_path = photo_path.parent / (stem + "_params.json")
        with open(params_in_path, "r") as file:
            params = json.load(file)

        calibration_mat = np.array(params["calibration_mat"])
        rotation_mat = np.array(params["x_rot_mat"]) @ np.array(params["z_rot_mat"]) @ np.array(params["pano_rot_mat"]).T
        translation = np.array(params["pano_translation"])
    else:
        K, R, T, _, _, _ = load_cameras_colmap(
            get_colmap_file(colmap_root, "images"), get_colmap_file(colmap_root, "cameras")
        )
        calibration_mat = K[index]
        rotation_mat = R[index]
        translation = (- R[index].T @ T[index]).squeeze()

    RGBcut, XYZcut, depth, _ = projectMesh(mesh, calibration_mat, rotation_mat, translation, False)

    sio.savemat(
        mat_out_path,
        {"RGBcut": cv2.imread(str(photo_path), cv2.IMREAD_UNCHANGED), "XYZcut": XYZcut}
    )
    sio.savemat(
        pose_out_path,
        {"R": rotation_mat, "position": translation, "calibration_mat": calibration_mat}
    )
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


if __name__ == "__main__":
    PREFIX = "/nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender"
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_root",
        type=Path,
        help="Root input data folder",
        default=f"{PREFIX}/2019-09-28_08.31.29/images/"
    )
    parser.add_argument(
        "--input_root_colmap", type=Path, help="colmap SfM output directory", default=None
    )
    parser.add_argument(
        "--input_ply_path",
        type=str,
        help="path to point cloud or mesh .ply file",
        default=f"{PREFIX}/2019-09-28_08.31.29/2019-09-28_08.31.29.ply"
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        help="path to write output data",
        default="/nfs/projects/artwin/experiments/artwin-inloc/2019-09-28_08.31.29"
    )
    parser.add_argument(
        "--test_size", type=int, default=0, help="Test size for generated dataset"
    )
    parser.add_argument(
        "--squarify",
        type=lambda x:bool(distutils.util.strtobool(x)),
        help="Should all images that fit be placed onto black canvas of size min_size x min_size?",
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

    ply_mesh = load_ply(args.input_ply_path)

    rnd = random.Random(42)

    model = args.output_root / "model" / "model_rotated.obj"
    if not model.exists():
        os.link(args.input_ply_path, model)

    if args.input_root_colmap is not None:
        _, _, _, Hs, _, img_nms = load_cameras_colmap(
            get_colmap_file(args.input_root_colmap, "images"),
            get_colmap_file(args.input_root_colmap, "cameras")
        )
        indices = list(range(len(Hs)))
        rnd.shuffle(indices)
        # Jump over the test set
        photos = [
            (
                args.input_root / img_nms[indices[i]],
                indices[i]
            ) for i in range(len(Hs)) if i >= args.test_size
        ]
    else:
        photos = list(args.input_root.glob("*.png"))
        photos = list(zip(photos, range(len(photos))))
    for idx, photo in enumerate(photos):
        path = photo[0]
        print(f"Processing {idx + 1}/{len(photos)}")
        cutoutFromPhoto(photo[1], ply_mesh, path, args.output_root, args.input_root_colmap)
