import mmint_utils
import numpy as np
import torch.utils.data
import torch
import os


class PokeDataset(torch.utils.data.Dataset):
    """
    Poke trial dataset. Not contact patch labels.
    """

    def __init__(self, dataset_dir: str, transform=None):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.transform = transform

        # Load dataset files and sort according to example number.
        data_fns = sorted([f for f in os.listdir(self.dataset_dir) if "out" in f and ".pkl.gzip" in f],
                          key=lambda x: int(x.split(".")[0].split("_")[-1]))
        self.num_trials = len(data_fns)

        # Data arrays.
        self.object_idcs = []  # Object index: what tool is used in this example?
        self.trial_idcs = []  # Trial index: which trial is used in this example?
        self.wrist_wrench = []  # Wrist wrench.
        self.partial_pointcloud = []  # Partial pointcloud.

        # Load all data.
        for trial_idx, data_fn in enumerate(data_fns):
            example_dict = mmint_utils.load_gzip_pickle(os.path.join(dataset_dir, data_fn))

            # Populate example info.
            self.object_idcs.append(0)  # TODO: Replace when using multiple tools.
            self.trial_idcs.append(trial_idx)
            self.wrist_wrench.append(np.array(example_dict["input"]["wrist_wrench"]))
            self.partial_pointcloud.append(example_dict["input"]["combined_pointcloud"][:, :3])

    def __len__(self):
        return self.num_trials

    def __getitem__(self, index):
        object_index = self.object_idcs[index]

        data_dict = {
            "object_idx": np.array([object_index]),
            "trial_idx": np.array([self.trial_idcs[index]]),
            "wrist_wrench": self.wrist_wrench[index],
            "partial_pointcloud": self.partial_pointcloud[index],
        }

        return data_dict