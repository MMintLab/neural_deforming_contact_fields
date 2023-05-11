import tqdm
import trimesh

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

    def __init__(self, dataset_dir: str, load_data: bool = True, transform=None, device="cpu"):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.dtype = torch.float32
        self.device = device

        # Load dataset files and sort according to example number.
        data_fns = sorted([f for f in os.listdir(self.dataset_dir) if "out" in f and ".pkl.gzip" in f],
                          key=lambda x: int(x.split(".")[0].split("_")[-1]))
        self.num_trials = len(data_fns)
        self.original_num_trials = len(data_fns)  # Above value may change due to bad data examples...

        nominal_fns = sorted([f for f in os.listdir(self.dataset_dir) if "nominal" in f],
                             key=lambda x: int(x.split(".")[0].split("_")[-1]))
        self.num_objects = len(nominal_fns)

        if not load_data:
            return

        # Data arrays.
        self.object_idcs = []  # Object index: what tool is used in this example?
        self.trial_idcs = []  # Trial index: which trial is used in this example?
        self.query_points = []  # Query points.
        self.sdf = []  # Signed distance value at point.
        self.normals = []  # Surface normals at point.
        self.in_contact = []  # Binary contact indicator at point.
        self.trial_pressure = []  # Contact force at point.
        self.wrist_wrench = []  # Wrist wrench.
        self.surface_points = []  # Surface points.
        self.surface_in_contact = []  # Surface point contact labels.
        self.partial_pointcloud = []  # Partial point cloud.
        self.contact_patch = []  # Contact patch.
        self.points_iou = []  # Points used to calculated IoU.
        self.occ_tgt = []  # Occupancy target for IoU points.

        # Load all data.
        for trial_idx, data_fn in enumerate(data_fns):
            example_dict = mmint_utils.load_gzip_pickle(os.path.join(dataset_dir, data_fn))

            # Populate example info.
            try:
                contact_patch = example_dict["test"]["contact_patch"]
            except:
                contact_patch = self.surface_points[-1][self.surface_in_contact[-1]]
            if len(contact_patch) == 0:
                self.num_trials -= 1
                continue
            self.contact_patch.append(contact_patch)

            self.object_idcs.append(0)  # TODO: Replace when using multiple tools.
            self.trial_idcs.append(trial_idx)
            self.query_points.append(example_dict["train"]["query_points"])
            self.sdf.append(example_dict["train"]["sdf"])
            self.normals.append(example_dict["train"]["normals"])
            self.in_contact.append(example_dict["train"]["in_contact"])
            self.trial_pressure.append(example_dict["train"]["pressure"])
            self.surface_points.append(example_dict["test"]["surface_points"])
            self.surface_in_contact.append(example_dict["test"]["surface_in_contact"])
            try:
                self.wrist_wrench.append(example_dict["input"]["wrist_wrench"])
            except:
                self.wrist_wrench.append(example_dict["train"]["wrist_wrench"])

            camera_idcs = [0, 1, 2]  # TODO: parameterize this somehow?
            partial_pc = []
            for camera_idx in camera_idcs:
                pc = example_dict["input"]["pointclouds"][camera_idx]["pointcloud"]
                if pc is not None:
                    partial_pc.append(pc)
            self.partial_pointcloud.append(np.concatenate(partial_pc, axis=0))

            self.points_iou.append(example_dict["test"]["points_iou"])
            self.occ_tgt.append(example_dict["test"]["occ_tgt"])

        # Load nominal geometry info.
        self.nominal_query_points = []
        self.nominal_sdf = []

        for object_idx, nominal_fn in enumerate(nominal_fns):
            example_dict = mmint_utils.load_gzip_pickle(os.path.join(dataset_dir, nominal_fn))

            # Populate nominal info.
            self.nominal_query_points.append(example_dict["query_points"])
            self.nominal_sdf.append(example_dict["sdf"])

        assert len(self.nominal_query_points) == max(self.object_idcs) + 1

    def get_num_objects(self):
        return self.num_objects

    def get_num_trials(self):
        return self.num_trials

    def get_example_mesh(self, example_idx):
        mesh_fn = os.path.join(self.dataset_dir, "out_%d_mesh.obj" % self.trial_idcs[example_idx])
        mesh = trimesh.load(mesh_fn)
        return mesh

    def __len__(self):
        return self.num_trials

    def __getitem__(self, index):
        object_index = self.object_idcs[index]

        data_dict = {
            "env_class": self.trial_idcs[index] // (self.original_num_trials // 3),  # NOTE: assumes equal num per env.
            "object_idx": np.array([object_index]),
            "trial_idx": np.array([self.trial_idcs[index]]),
            "query_point": self.query_points[index],
            "sdf": self.sdf[index],
            "normals": self.normals[index],
            "in_contact": self.in_contact[index].astype(int),
            "pressure": np.array([self.trial_pressure[index]]),
            "wrist_wrench": self.wrist_wrench[index],
            "nominal_query_point": self.nominal_query_points[object_index],
            "nominal_sdf": self.nominal_sdf[object_index],
            "surface_points": self.surface_points[index],
            "surface_in_contact": self.surface_in_contact[index],
            "partial_pointcloud": self.partial_pointcloud[index],
            "contact_patch": self.contact_patch[index],
            "points_iou": self.points_iou[index],
            "occ_tgt": self.occ_tgt[index],
        }

        if self.transform is not None:
            data_dict = self.transform(data_dict)

        return data_dict
