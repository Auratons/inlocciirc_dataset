# pylint: disable=wrong-import-position,missing-module-docstring,missing-function-docstring
import argparse
import json
import os
from pathlib import Path

import numpy as np
import scipy.io as sio

import cv2


def get_central_crop(img, crop_height=512, crop_width=512):
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
    assert len(img.shape) == 3, (
        "input image should be either a 2D or 3D matrix,"
        " but input was of shape %s" % str(img.shape)
    )
    height, width, _ = img.shape
    assert height >= crop_height and width >= crop_width, (
        "input image cannot " "be smaller than the requested crop size"
    )
    st_y = (height - crop_height) // 2
    st_x = (width - crop_width) // 2
    return np.squeeze(img[st_y : st_y + crop_height, st_x : st_x + crop_width, :])


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
    points[points.sum(axis=2) < 0.005] = np.nan
    points = points + t.reshape((1,1,3))
    return points

# Renderer is used for unifying depth semantics (there was a lot of already
# generated data, so it was easier to do workaround for the pre-generated).
def cutoutFromPhoto(photo_path, output_root, local_to_global, matrices, renderer_type, lift_pcd):
    stem = photo_path.stem.replace("_reference", "")

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

    if mesh_out_path.exists():
        return

    calibration_mat = np.array(matrices["train"][str(mesh_projection)]["calibration_mat"])
    camera_pose = np.array(matrices["train"][str(mesh_projection)]["camera_pose"])

    camera_position = camera_pose[:3, 3]
    camera_orientation = camera_pose[:3, :3]

    depth = np.load(str(depth_npy))
    if renderer_type == "pyrender":
        camera_orientation[:, 1:3] *= -1
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
        camera_orientation[:, 1:3] *= -1
        depth[np.abs(depth - 1) < (np.max(depth) / 10)] = 0.0
    elif renderer_type == "splatter":
        # In the latest code, we get the right z-depth, fixing just pose.
        camera_orientation[:, 1:3] *= -1
    XYZcut = compute_xyz_cut(calibration_mat, camera_orientation, camera_position, depth)

    # Move floor pcd to virtual global coordinate system to disambiguate localization.
    floor_num = int(photo_path.parent.parent.name[-1])
    if lift_pcd and floor_num > 1:
        print(f"Moving {photo_path.parent.parent.name} pcd higher by {(floor_num -1) * 40}.")
        XYZcut[:, :, 2] += (floor_num -1) * 40.0

    prj = cv2.imread(str(mesh_projection), cv2.IMREAD_UNCHANGED)[:, :, :3]
    ref = cv2.imread(str(cutout_reference), cv2.IMREAD_UNCHANGED)[:, :, :3]
    assert prj.shape[:2] == ref.shape[:2]  #  == depth.shape[:2]

    sio.savemat(
        mat_out_path,
        {"RGBcut": prj, "XYZcut": XYZcut}
    )
    sio.savemat(
        pose_out_path,
        {"R": camera_orientation, "position": camera_position, "calibration_mat": calibration_mat}
    )
    if not cutout_out_path.exists():
        # cv2.imwrite(str(cutout_out_path), ref)
        os.link(cutout_reference, cutout_out_path)

    # For possible further recalculation, npz with raw depth map is also saved.
    if not Path(str(depth_out_path) + ".npy").exists():
        os.link(depth_npy, str(depth_out_path) + ".npy")
    if not mesh_out_path.exists():
        os.link(mesh_projection, mesh_out_path)

def get_path_name(building):
    if "CSE" in building:
        return "cse"
    if "DUC" in building:
        return "DUC"

def load_initial_transform(transform_path):
    """Load the transformation matrix contained in the `xxx_trans_xxx.txt` file."""
    transform = []
    with open(transform_path, "r") as f:
        for line in f.readlines()[7:-1]:
            transform.append([float(x) for x in line.split()])
    return np.array(transform, dtype=np.float)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inloc_path", type=Path, default="/home/kremeto1/neural_rendering/datasets/raw/inloc"
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        help="Root input data folder",
        default="/home/kremeto1/neural_rendering/datasets/processed/inloc/inloc_rendered_pyrender"
    )
    parser.add_argument(
        "--input_root_renderer",
        type=str,
        help="One of 'pyrender', 'splatter', 'marcher'.",
        default="pyrender"
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        help="path to write output data",
        default="/home/kremeto1/neural_rendering/datasets/final/inloc/inloc_rendered_pyrender_inloc_format"
    )
    parser.add_argument(
        "--lift",
        action='store_true',
        help="Separate floors by vertical distance?",
    )
    parser.set_defaults(lift=False)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "depthmaps").mkdir(exist_ok=True)
    (args.output_root / "meshes").mkdir(exist_ok=True)
    (args.output_root / "cutouts").mkdir(exist_ok=True)
    (args.output_root / "matfiles").mkdir(exist_ok=True)
    (args.output_root / "poses").mkdir(exist_ok=True)

    for building in args.input_root.iterdir():
        print("Building dataset for {}".format(building.name))
        for scan in (args.input_root / building.name).iterdir():

            if not scan.is_dir():
                continue

            matrices_file = scan / "matrices_for_rendering.txt"
            with open(matrices_file, "r") as file:
                matrices = json.load(file)

            # transform_path = (
            #     args.inloc_path
            #     / f"database/alignments/{building}/transformations"
            #     / f"{get_path_name(building.name)}_trans_{scan}.txt"
            # )
            # T = load_initial_transform(transform_path)

            print(f"Processing {scan.name}")
            for idx, photo_path in enumerate(scan.glob("*_reference.png")):
                print(f"Processing {idx + 1}/36")
                cutoutFromPhoto(photo_path, args.output_root, None, matrices, args.input_root_renderer, args.lift)
