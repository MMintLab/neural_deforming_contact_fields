import numpy as np
import torch.utils.data
import torch
from neural_contact_fields.data import dataset_helpers


class SingleToolDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_fn: str, split: str, tool_idx: int, deformation_idx: int, transform=None):
        super().__init__()
        self.dataset_fn = dataset_fn
        self.split = split
        self.tool_idx = tool_idx
        self.deformation_idx = deformation_idx
        self.transform = transform

        # Load full dataset.
        dataset_dict = dataset_helpers.load_dataset_dict(dataset_fn)

        # Pull out specific tool deformation info.
        self.data_dict = dataset_dict[split][tool_idx][deformation_idx]

        # Coordinate query points.
        self.coords = self.data_dict["coords"][0]
        self.sdf = self.data_dict["gt"][0]
        self.normals = self.data_dict["normals"][0]
        self.contact = torch.zeros([self.coords.shape[0], 1], dtype=torch.float32)

        # Add the contact points to the points.
        contact_points = self.data_dict["contact"][0]
        contact_sdf = torch.zeros([contact_points.shape[0], 1], dtype=torch.float32)
        contact = torch.ones([contact_points.shape[0], 1], dtype=torch.float32)
        self.coords = torch.cat([self.coords, contact_points], dim=0)
        self.sdf = torch.cat([self.sdf, contact_sdf], dim=0)
        self.contact = torch.cat([self.contact, contact], dim=0)

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        data_dict = {
            "coord": self.coords[idx].float(),
            "sdf": self.sdf[idx].float(),
            "contact": self.contact[idx].float(),
        }

        return data_dict
