import numpy as np
import transforms3d as tf3d


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


def transform_pointcloud(pointcloud: np.ndarray, transform: np.ndarray):
    points_homogenous = np.ones(shape=[pointcloud.shape[0], 4])
    points_homogenous[:, :3] = pointcloud

    tf_pc = (transform @ points_homogenous.T).T[:, :3]
    return tf_pc
