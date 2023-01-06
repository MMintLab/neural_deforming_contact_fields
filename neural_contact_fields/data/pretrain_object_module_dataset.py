import os

import torch.utils.data
import torch
import mmint_utils
import numpy as np


class PretrainObjectModuleDataset(torch.utils.data.Dataset):
    """
    Each example in this dataset is a full object's query points/sdf labels.
    """

    # TODO: Version that lets us handle multiple tools.

    def __init__(self, dataset_fn: str, transform=None):
        super().__init__()
        self.dataset_fn = dataset_fn
        self.transform = transform
        self.num_objects = 1

        data_dict = mmint_utils.load_gzip_pickle(self.dataset_fn)

        # Load dataset files and sort according to example number.
        data_fns = sorted([f for f in os.listdir(os.path.dirname(self.dataset_fn)) if "out" in f],
                          key=lambda x: int(x.split(".")[0].split("_")[-1]))
        self.num_trials = len(data_fns)

        self.object_idcs = [0]
        self.n_points = [data_dict["n_points"]]
        self.query_points = [data_dict["query_points"]]
        self.normals = [data_dict["normals"]]
        self.sdf = [data_dict["sdf"]]

    def __len__(self):
        return self.num_objects

    def __getitem__(self, index):
        data_dict = {
            "object_idx": np.array([self.object_idcs[index]]),
            "query_point": self.query_points[index].astype(float),
            "sdf": self.sdf[index].astype(float),
            "normals": self.normals[index].astype(float),
        }

        return data_dict
