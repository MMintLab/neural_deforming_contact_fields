import pdb

import mmint_utils
import numpy as np
import torch.utils.data
import torch
import os


class ToolDataset(torch.utils.data.Dataset):
    """
    Tool dataset. Contains query points and SDF, binary contact, and contact force at each point.
    Each example in this dataset is all sample points for a given trial.
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
        self.object_idcs = []  # Object index: what tool is used in this example?
        self.trial_idcs = []  # Trial index: which trial is used in this example?
        self.query_points = []  # Query points.
        self.sdf = []  # Signed distance value at point.
        self.in_contact = []  # Binary contact indicator at point.
        self.trial_pressure = []  # Contact force at point.

        # Load all data.
        for trial_idx, data_fn in enumerate(data_fns):
            example_dict = mmint_utils.load_gzip_pickle(os.path.join(dataset_dir, data_fn))

            # Populate example info.
            self.object_idcs.append(0)  # TODO: Replace when using multiple tools.
            self.trial_idcs.append(trial_idx)
            self.query_points.append(example_dict["query_points"])
            self.sdf.append(example_dict["sdf"])
            self.in_contact.append(example_dict["in_contact"])
            self.trial_pressure.append(example_dict["pressure"])

    def __len__(self):
        return len(self.object_idcs)

    def __getitem__(self, index):
        data_dict = {
            "object_idx": self.object_idcs[index],
            "trial_idx": self.trial_idcs[index],
            "query_point": self.query_points[index],
            "sdf": self.sdf[index],
            "in_contact": self.in_contact[index].astype(int),
            "pressure": self.trial_pressure[index]
        }

        return data_dict
