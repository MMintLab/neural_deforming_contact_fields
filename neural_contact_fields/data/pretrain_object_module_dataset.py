import torch.utils.data
import torch
import mmint_utils


class PretrainObjectModuleDataset(torch.utils.data.Dataset):
    """
    Each example in this dataset is a full object's query points/sdf labels.
    """

    # TODO: Version that lets us handle multiple tools.

    def __init__(self, dataset_fn: str, transform=None):
        super().__init__()
        self.dataset_fn = dataset_fn
        self.transform = transform
        self.num_trials = 1

        data_dict = mmint_utils.load_gzip_pickle(self.dataset_fn)
        self.n_points = [data_dict["n_points"]]
        self.query_points = [data_dict["query_points"]]
        self.normals = [data_dict["normals"]]
        self.sdf = [data_dict["sdf"]]

    def __len__(self):
        return self.num_trials

    def __getitem__(self, index):
        data_dict = {
            "query_point": self.query_points[index],
            "sdf": self.sdf[index],
            "normals": self.normals[index],
        }

        return data_dict
