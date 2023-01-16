import os

import mmint_utils
import numpy as np
import torch.utils.data
import trimesh
from neural_contact_fields.utils import utils


class RealToolDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_dir: str, transform=None):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.transform = transform

        # Load dataset files and sort according to example number.
        data_fns = sorted([f for f in os.listdir(self.dataset_dir) if "config" in f and ".pkl.gzip" in f],
                          key=lambda x: int(x.split(".")[0].split("_")[-1]))
        self.num_trials = len(data_fns)

        self.object_idcs = []
        self.surface_points = []
        self.wrist_wrench = []
        self.query_point = []

        for trial_idx, data_fn in enumerate(data_fns):
            example_dict = mmint_utils.load_gzip_pickle(os.path.join(dataset_dir, data_fn))

            self.object_idcs.append(0)  # TODO: Replace when using multiple tools.

            # Load wrist wrench data.
            real_wrench = np.array(example_dict["tactile"]["ati_wrench"][-1][0])
            self.wrist_wrench.append(real_wrench)

            # Load surface_points.
            surface_points = trimesh.load(os.path.join(dataset_dir, "config_%d_photoneo_surface.ply" % trial_idx))
            surface_points = np.array(surface_points.vertices)

            # Transform surface points to tool frame.
            ee_pose = np.concatenate(example_dict["proprioception"]["ee_pose"][0], axis=0)
            tf_matrix = utils.pose_to_matrix(utils.xyzw_to_wxyz(ee_pose))
            surface_points = utils.transform_pointcloud(surface_points, np.linalg.inv(tf_matrix))
            self.surface_points.append(surface_points)

            self.query_point.append(np.zeros([5, 3]))

    def __len__(self):
        return self.num_trials

    def __getitem__(self, idx):
        return {
            "object_idx": np.array([self.object_idcs[idx]]),
            "surface_points": self.surface_points[idx],
            "wrist_wrench": self.wrist_wrench[idx],
            "query_point": self.query_point[idx],
        }
