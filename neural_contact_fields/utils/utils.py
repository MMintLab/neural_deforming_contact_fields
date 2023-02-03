import numpy as np
import torch
import transforms3d as tf3d
import open3d as o3d


def pointcloud_to_o3d(pointcloud):
    """
    Send pointcloud to open3d.
    """
    pointcloud_xyz = pointcloud[:, :3]
    if pointcloud.shape[1] == 4:
        color = np.zeros([pointcloud.shape[0], 3], dtype=float)
        color[:, 0] = pointcloud[:, 3]
        color[:, 1] = pointcloud[:, 3]
        color[:, 2] = pointcloud[:, 3]
    elif pointcloud.shape[1] > 4:
        color = pointcloud[:, 3:]
    else:
        color = None

    points = o3d.geometry.PointCloud()
    points.points = o3d.utility.Vector3dVector(pointcloud_xyz)
    if color is not None:
        points.colors = o3d.utility.Vector3dVector(color)
    return points


def transform_pointcloud(pointcloud, transform):
    """
    Transform the given pointcloud by the given matrix transformation T.
    """
    pointcloud_pcd: o3d.geometry.PointCloud = pointcloud_to_o3d(pointcloud)
    pointcloud_pcd.transform(transform)
    if pointcloud.shape[1] > 3:
        return np.concatenate([np.asarray(pointcloud_pcd.points), pointcloud[:, 3:]], axis=1)
    else:
        return np.asarray(pointcloud_pcd.points)


def sample_pointcloud(pointcloud, n):
    """
    Draw n samples from given pointcloud.
    """
    pointcloud_size = len(pointcloud)
    indices = np.arange(pointcloud_size)
    sample_indices = np.random.choice(indices, size=n)

    return pointcloud[sample_indices]


def save_pointcloud(pointcloud, fn: str):
    pointcloud_pcd: o3d.geometry.PointCloud = pointcloud_to_o3d(pointcloud)
    o3d.io.write_point_cloud(fn, pointcloud_pcd)


def load_pointcloud(fn: str):
    pointcloud_pcd = o3d.io.read_point_cloud(fn)

    if pointcloud_pcd.has_colors():
        return np.concatenate([np.asarray(pointcloud_pcd.points), np.asarray(pointcloud_pcd.colors)], axis=1)
    else:
        return np.asarray(pointcloud_pcd.points)


def pose_to_matrix(pose, axes="rxyz"):
    """
    Pose to matrix.
    """
    matrix = np.eye(4, dtype=pose.dtype)
    matrix[:3, 3] = pose[:3]

    if len(pose) == 6:
        matrix[:3, :3] = tf3d.euler.euler2mat(pose[3], pose[4], pose[5], axes=axes)
    else:
        matrix[:3, :3] = tf3d.quaternions.quat2mat(pose[3:])

    return matrix


def matrix_to_pose(matrix, axes="rxyz"):
    """
    Matrix to pose.
    """
    pose = np.zeros(6)
    pose[:3] = matrix[:3, 3]
    pose[3:] = tf3d.euler.mat2euler(matrix, axes=axes)
    return pose


def xyzw_to_wxyz(pose):
    """
    Convert pose with xyzw quat convention to wxyz convention.
    """
    quat = pose[3:]
    new_quat = np.array([quat[3], quat[0], quat[1], quat[2]])
    pose[3:] = new_quat
    return pose


def numpy_dict(torch_dict: dict):
    np_dict = dict()
    for k, v in torch_dict.items():
        if type(v) is torch.Tensor:
            np_dict[k] = v.detach().cpu().numpy()
        else:
            np_dict[k] = v
    return np_dict
