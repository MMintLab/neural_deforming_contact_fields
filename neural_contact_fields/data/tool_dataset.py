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
        self.normals = []  # Surface normals at point.
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
            self.normals.append(example_dict["normals"])
            self.in_contact.append(example_dict["in_contact"])
            self.trial_pressure.append(example_dict["pressure"])

        self.num_objects = max(self.object_idcs) + 1

        # Load nominal geometry info.
        self.nominal_query_points = []
        self.nominal_sdf = []
        nominal_fns = sorted([f for f in os.listdir(self.dataset_dir) if "nominal" in f],
                             key=lambda x: int(x.split(".")[0].split("_")[-1]))

        for object_idx, nominal_fn in enumerate(nominal_fns):
            example_dict = mmint_utils.load_gzip_pickle(os.path.join(dataset_dir, nominal_fn))

            # Populate nominal info.
            self.nominal_query_points.append(example_dict["query_points"])
            self.nominal_sdf.append(example_dict["sdf"])

    def get_num_objects(self):
        return self.num_objects

    def get_num_trials(self):
        return self.num_trials

    def __len__(self):
        return self.num_trials

    def __getitem__(self, index):
        object_index = self.object_idcs[index]

        data_dict = {
            "object_idx": np.array([object_index]),
            "trial_idx": np.array([self.trial_idcs[index]]),
            "query_point": self.query_points[index],
            "sdf": self.sdf[index],
            "normals": self.normals[index],
            "in_contact": self.in_contact[index].astype(int),
            "pressure": np.array([self.trial_pressure[index]]),
            "nominal_query_point": self.nominal_query_points[object_index],
            "nominal_sdf": self.nominal_sdf[object_index],
        }

        return data_dict
