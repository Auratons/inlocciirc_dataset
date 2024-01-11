# pylint: disable=wrong-import-position,missing-module-docstring,missing-function-docstring
import os
import sys
from pathlib import Path

sys.path.insert(
    0,
    str((Path(__file__).parent / "../../functions/inLocCIIRC_utils/projectMesh").resolve())
)

import argparse
import json

import numpy as np
import scipy.io as sio

import cv2
import open3d as o3d
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
    points[points.sum(axis=2) < 0.005] = np.nan
    points = points + t.reshape((1,1,3))
    return points

# Renderer is used for unifying depth semantics (there was a lot of already
# generated data, so it was easier to do workaround for the pre-generated).
def cutoutFromPhoto(mapping, photo_path, output_root, renderer_type):
    stem = photo_path.stem.replace("_reference", "")

    source_photo = Path(mapping[str(photo_path)])
    matrices_file = source_photo.parent.parent / "train" / "matrices_for_rendering.json"
    params_in_path = source_photo.parent / (source_photo.stem.strip("_reference") + "_params.json")

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

    try:
        with open(matrices_file, "r") as file:
            params = json.load(file)
        calibration_mat = np.array(params["train"][str(source_photo.parent / (source_photo.stem.strip("_reference") + "_color.png"))]["calibration_mat"])
        camera_pose = np.array(params["train"][str(source_photo.parent / (source_photo.stem.strip("_reference") + "_color.png"))]["camera_pose"])
    except:
        with open(params_in_path, "r") as file:
            params = json.load(file)
        calibration_mat = np.array(params["calibration_mat"])
        camera_pose = np.array(params["camera_pose"])

    camera_position = camera_pose[:3, 3]
    camera_orientation = camera_pose[:3, :3]

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
        camera_orientation[:, 1:3] *= -1
        depth[np.abs(depth - 1) < (np.max(depth) / 10)] = 0.0
    elif renderer_type == "splatter":
        # In the latest code, we get the right z-depth, fixing just pose.
        camera_orientation[:, 1:3] *= -1
    XYZcut = compute_xyz_cut(calibration_mat, camera_orientation, camera_position, depth)

    if "53" in source_photo.parent.parent.name:  # Move hall 53 to virtual global coordinate system to disambiguate localization.
        print("Moving pcd higher.")
        XYZcut[:, :, 2] += 50.0

    prj = cv2.imread(str(mesh_projection), cv2.IMREAD_UNCHANGED)[:, :, :3]
    ref = cv2.imread(str(cutout_reference), cv2.IMREAD_UNCHANGED)[:, :, :3]
    assert prj.shape[:2] == ref.shape[:2]  #  == depth.shape[:2]

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_root",
        type=Path,
        help="Root input data folder",
        default="/nfs/projects/artwin/experiments/hololens_mapper/joined_dataset"
    )
    parser.add_argument(
        "--input_root_renderer",
        type=str,
        help="One of 'pyrender', 'splatter', 'marcher'.",
        default="pyrender"
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

    # # Get roots of nriw training datasets from which a joined dataset was generated.
    # source_roots = {}
    # for path in [Path(k).parent.parent for k in sub_mapping_1.values()]:
    #     source_roots[str(path)] = 1
    # source_roots = list(source_roots.keys())

    # # Get mapping from partial nriw training datasets to source references
    # # generated by matlab from artwin panoramas
    # sub_mappings_2 = []
    # for mp in [Path(root) / "mapping.txt" for root in source_roots]:
    #     with open(mp, "r") as file:
    #         lines = file.readlines()
    #         # Filter lines only for used source (train/val/test)
    #         lines = filter(lambda line: str(args.input_root.name).upper() in line, lines)
    #         # Get rid of trailing TRAIN/DEV/TEST
    #         lines = [str.join(" ", line.split(" ")[:-1]) for line in lines]
    #         line_tuples = [tuple(line.split(" -> ")) for line in lines]  # (source, dest)
    #         sub_map = {}
    #         for wut in zip(line_tuples):
    #             v, k = wut[0]
    #             sub_map[str(Path(mp).parent / str(args.input_root.name) / f"{int(k):04n}_reference.png")] = f"/nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/{str.join('_', v.split('_')[0:2])}/images/{v}"
    #         sub_mappings_2.append(sub_map)

    # # Get mapping from reference images in joined dataset for nriw training to
    # # source references generated by matlab from artwin panoramas
    # mapping = {}
    # for k, v in sub_mapping_1.items():
    #     for sm in sub_mappings_2:
    #         if v in sm:
    #             mapping[k] = sm[v]

    photos = list((args.input_root / "val").glob("*_reference.png")) + list((args.input_root / "train").glob("*_reference.png"))
    for idx, photo_path in enumerate(photos):
        print(f"Processing {idx + 1}/{len(photos)}")
        cutoutFromPhoto(sub_mapping_1, photo_path, args.output_root, args.input_root_renderer)
