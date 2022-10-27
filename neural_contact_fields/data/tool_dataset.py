import mmint_utils
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
        data_fns = sorted(os.listdir(self.dataset_dir), key=lambda x: int(x.split(".")[0].split("_")[-1]))
        self.num_trials = len(data_fns)

        # Data arrays.
        self.object_idcs = []  # Object index: what tool is used in this example?
        self.trial_idcs = []  # Trial index: which trial is used in this example?
        self.query_points = []  # Query points.
        self.sdf = []  # Signed distance value at point.
        self.in_contact = []  # Binary contact indicator at point.
        self.forces = []  # Contact force at point.

        # Load all data.
        for trial_idx, data_fn in enumerate(data_fns):
            example_dict = mmint_utils.load_gzip_pickle(data_fn)
            n_points = example_dict["n_points"]

            # Populate example info.
            self.object_idcs.extend([0] * n_points)  # TODO: Replace when using multiple tools.
            self.trial_idcs.extend([trial_idx] * n_points)
            self.query_points.extend(example_dict["query_points"])
            self.sdf.extend(example_dict["sdf"])
            self.in_contact.extend(example_dict["in_contact"])
            self.forces.extend(example_dict["forces"])

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
