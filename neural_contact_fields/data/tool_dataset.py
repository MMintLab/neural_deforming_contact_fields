import pdb

import mmint_utils
import numpy as np
import torch.utils.data
import torch
import os


class ToolDataset(torch.utils.data.Dataset):
    """
    Tool dataset. Contains query points and SDF, binary contact, and contact force at each point.
    """

    def __init__(self, dataset_dir: str, transform=None):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.transform = transform

        # Load dataset files and sort according to example number.
        data_fns = sorted([f for f in os.listdir(self.dataset_dir) if "out" in f],
                          key=lambda x: int(x.split(".")[0].split("_")[-1]))
        self.num_trials = len(data_fns)

        # Data arrays.
        self.object_idcs = np.empty(0)  # Object index: what tool is used in this example?
        self.trial_idcs = np.empty(0)  # Trial index: which trial is used in this example?
        self.query_points = np.empty([0, 3])  # Query points.
        self.sdf = np.empty(0)  # Signed distance value at point.
        self.in_contact = np.empty(0)  # Binary contact indicator at point.
        self.forces = np.empty([0, 3])  # Contact force at point.

        # Load all data.
        for trial_idx, data_fn in enumerate(data_fns):
            example_dict = mmint_utils.load_gzip_pickle(os.path.join(dataset_dir, data_fn))
            n_points = example_dict["n_points"]

            # Populate example info.
            self.object_idcs = np.concatenate((self.object_idcs,
                                               [0] * n_points))  # TODO: Replace when using multiple tools.
            self.trial_idcs = np.concatenate((self.trial_idcs, [trial_idx] * n_points))
            self.query_points = np.concatenate((self.query_points, example_dict["query_points"]))
            self.sdf = np.concatenate((self.sdf, example_dict["sdf"]))
            self.in_contact = np.concatenate((self.in_contact, example_dict["in_contact"]))
            self.forces = np.concatenate((self.forces, example_dict["forces"]))

    def __len__(self):
        return len(self.object_idcs)

    def __getitem__(self, index):
        data_dict = {
            "object_idx": self.object_idcs[index],
            "trial_idx": self.trial_idcs[index],
            "query_point": self.query_points[index],
            "sdf": self.sdf[index],
            "in_contact": self.in_contact[index],
            "force": self.forces[index]
        }

        return data_dict

    def get_trial_indices(self):
        """
        Return all the unique trial indices for this dataset.
        # TODO: Handling different object indices.
        """
        return np.arange(self.num_trials)

    def get_all_points_for_trial(self, object_idx, trial_idx):
        # TODO: Handle for different object indices as well.
        trial_idcs = np.nonzero(self.trial_idcs == trial_idx)[0].astype(int)

        data_dict = {
            "object_idx": object_idx,
            "trial_idx": trial_idx,
            "query_point": self.query_points[trial_idcs],
            "sdf": self.sdf[trial_idcs],
            "in_contact": self.in_contact[trial_idcs],
            "force": self.forces[trial_idcs]
        }
        return data_dict
