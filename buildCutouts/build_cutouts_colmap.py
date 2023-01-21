# pylint: disable=wrong-import-position,missing-module-docstring,missing-function-docstring,no-member,c-extension-no-member
import os
import sys
from pathlib import Path

sys.path.insert(
    0,
    str((Path(__file__).parent / "../../functions/inLocCIIRC_utils/projectMesh").resolve())
)

import argparse
import distutils
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

import cv2
import open3d as o3d
import read_model
from projectMesh import buildXYZcut


def project_mesh_core(depth, k, R, t, debug):
    sensor_width = depth.shape[1]
    sensor_height = depth.shape[0]
    focal_length = (k[0, 0] + k[1, 1]) / 2
    scaling = 1.0 / focal_length
    space_coord_system = np.eye(3)
    sensor_coord_system = np.matmul(R, space_coord_system)
    sensor_x_axis = sensor_coord_system[:, 0]
    sensor_y_axis = sensor_coord_system[:, 1]
    camera_dir = sensor_coord_system[:, 2]

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

def compute_xyz_cut(k, R, t, depth):
    """
    For a 2D rectangle of RGB values computes the respective 2D rectangle of 3D points (thus XYZcut).
    It ensures that a pixel at any 2D position and 3D point in the resulting cut on the same position
    refer to the same physical point in the pointcloud.

    This can be still made nicer, but not enough time... Checking was done with matching pcd generated
    from a cut to the global source pcd. For checking spatial coherency across 2D rectangle, color gradient
    was used.

    ```
    points = sio.loadmat(XYZcut_path)["XYZcut"]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.reshape((-1,3)))
    pcd.colors = o3d.utility.Vector3dVector(((np.clip(cv2.applyColorMap((np.tile(np.arange(points.shape[1]),(points.shape[0],1))*255/points.shape[1]).astype(np.uint8), cv2.COLORMAP_HSV),0,255))/255).reshape((-1,3)))
    o3d.io.write_point_cloud('test.ply', pcd)
    ```
    """
    assert k[0, 0] == k[1, 1]
    focal_length = k[0, 0]
    hh = k[1, 2]  # half height
    hw = k[0, 2]  # half width
    shape = (int(2 * hh), int(2 * hw))

    pixel_centers = np.append(np.transpose(np.mgrid[-hw:hw, -hh:hh], (2, 1, 0)) / focal_length, np.ones(shape + (1,)), axis=2)
    points = np.matmul(np.tile(R, shape + (1, 1)), pixel_centers[:, :, :, np.newaxis]).squeeze()[:, :, :3]
    points = np.multiply(points, np.repeat(depth[:, :, np.newaxis], 3, axis=2))
    points = points + t.reshape((1,1,3))
    return points

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

# Renderer is used for unifying depth semantics (there was a lot of already
# generated data, so it was easier to do workaround for the pre-generated).
def cutoutFromPhoto(calibration_mat, rotation_mat, translation, photo_path, output_root, square, renderer_type):
    stem = photo_path.stem.strip("_reference")

    depth_npy = photo_path.parent / (stem + "_depth.npy")
    if not depth_npy.exists():
        depth_npy = photo_path.parent / (stem + "_depth.png.npy")
    mesh_projection = photo_path.parent / (stem + "_color.png")
    cutout_reference = photo_path

    mesh_out_path = output_root / "meshes" / ("mesh_" + stem + ".png")
    cutout_out_path = output_root / "cutouts" / ("cutout_" + stem + ".png")
    mat_out_path = output_root / "matfiles" / (cutout_out_path.name + ".mat")
    pose_out_path = output_root / "poses" / (cutout_out_path.name + ".mat")
    depth_out_path = output_root / "depthmaps" / ("depth_" + stem + ".png")

    # if mesh_out_path.exists():
    #     return

    # COLMAP stores view matrices, XYZcut expects camera matrix
    camera_position = (- rotation_mat.T @ translation).squeeze()
    camera_orientation = rotation_mat.T
    depth = np.load(str(depth_npy))
    if renderer_type == "pyrender":
        pass  # Real depth correctly recomputed by its internals.
    elif renderer_type == "marcher":
        hh = calibration_mat[1, 2]
        hw = calibration_mat[0, 2]
        f = calibration_mat[0, 0]
        assert calibration_mat[0, 0] == calibration_mat[1, 1], "Camera pixel is not square."
        depth = np.divide(
            depth,
            np.sqrt(np.square(np.transpose(np.mgrid[-hh:hh, -hw:hw], (1, 2, 0)) / f).sum(axis=2) + 1)
        )  # Marcher uses real distance from camera center instead of z depth, this fixes it.
    elif renderer_type == "splatter":
        pass  # In the latest code, we get the right z-depth.
    XYZcut = compute_xyz_cut(calibration_mat, camera_orientation, camera_position, depth)

    prj = cv2.imread(str(mesh_projection), cv2.IMREAD_UNCHANGED)[:, :, :3]
    ref = cv2.imread(str(cutout_reference), cv2.IMREAD_UNCHANGED)[:, :, :3]
    # print(f"Color: {prj.shape}")
    # print(f"Depth: {depth.shape}")
    # print(f"Ref: {ref.shape}")
    assert prj.shape[:2] == ref.shape[:2]  #  == depth.shape[:2]

    if square:
        XYZcut = squarify(XYZcut, ref.shape[0])

    sio.savemat(
        mat_out_path,
        {"RGBcut": prj, "XYZcut": XYZcut}
    )
    # np.save(str(mat_out_path) + ".npy", XYZcut)
    sio.savemat(
        pose_out_path,
        {"R": camera_orientation, "position": camera_position, "calibration_mat": calibration_mat}
    )
    if not cutout_out_path.exists():
        os.link(cutout_reference, cutout_out_path)

    # Debug
    # plt.imsave(depth_out_path, depth, cmap=plt.cm.gray_r)
    # cv2.imwrite(
    #     str(depth_out_path.parent / (depth_out_path.stem + "_uint16.png")),
    #     depth.astype(np.uint16),
    # )
    # For possible further recalculation, npz with raw depth map is also saved.
    if not Path(str(depth_out_path) + ".npy").exists():
        os.link(depth_npy, str(depth_out_path) + ".npy")
    if not mesh_out_path.exists():
        os.link(mesh_projection, mesh_out_path)

def squarify(image: np.array, square_size: int) -> np.array:
    assert square_size >= image.shape[0] and square_size >= image.shape[1]
    shape = list(image.shape)
    shape[0] = shape[1] = square_size
    square = np.zeros(shape, dtype=image.dtype)
    h, w = image.shape[:2]
    offset_h = (square_size - h) // 2
    offset_w = (square_size - w) // 2
    square[offset_h : offset_h + h, offset_w : offset_w + w, ...] = image
    return square

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
        "--input_root_renderer",
        type=str,
        help="One of 'pyrender', 'splatter', 'marcher'.",
        default="pyrender"
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
        default=False,
        help="Should all images that fit be placed onto black canvas of size min_size x min_size?",
    )
    parser.add_argument(
        "--min_size", type=int, default=512, help="Minimum size for images"
    )
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Train/val ratio")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "model").mkdir(exist_ok=True)
    (args.output_root / "sweepData").mkdir(exist_ok=True)
    (args.output_root / "depthmaps").mkdir(exist_ok=True)
    (args.output_root / "meshes").mkdir(exist_ok=True)
    (args.output_root / "cutouts").mkdir(exist_ok=True)
    (args.output_root / "matfiles").mkdir(exist_ok=True)
    (args.output_root / "poses").mkdir(exist_ok=True)

    rnd = random.Random(42)

    model = args.output_root / "model" / "model_rotated.obj"
    if not model.exists():
        os.link(args.input_ply_path, model)

    Ks, Rs, Ts, Hs, Ws, img_nms = load_cameras_colmap(
        get_colmap_file(args.input_root_colmap, "images"),
        get_colmap_file(args.input_root_colmap, "cameras")
    )
    indices = list(range(len(Hs)))
    rnd.shuffle(indices)

    split = int(args.val_ratio * (len(Hs) - args.test_size))
    it = -1

    for idx in range(len(Hs)):
        print(f"Processing {idx + 1}/{len(Hs)}")
        i = indices[idx]
        if args.squarify or min(Ws[i], Hs[i]) > args.min_size:
            it += 1
            # Jump over the test set
            if it < args.test_size:
                continue
            elif it < args.test_size + split:
                dir = args.input_root / "val"
            else:
                dir = args.input_root / "train"

            cutoutFromPhoto(
                Ks[i],
                Rs[i],
                Ts[i],
                dir / "{:04n}_reference.png".format(it),
                args.output_root,
                args.squarify,
                args.input_root_renderer
            )

        else:
            print("Skipping this image : too small ")
