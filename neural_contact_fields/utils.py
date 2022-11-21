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
