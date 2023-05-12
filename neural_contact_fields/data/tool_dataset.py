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
        # data_fns = data_fns[:10]
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
        self.partial_pointcloud = []  # Partial pointcloud.
        self.contact_patch = []  # Contact patch.
        self.points_iou = []  # Points used to calculated IoU.
        self.occ_tgt = []  # Occupancy target for IoU points.
        # self.partial_pcd_idx = [[0, 1, 7]]
        self.partial_pcd_idx = [[3, 4, 5]]
        # self.partial_pcd_idx = [[0], [1], [2], [3], [4], [5], [6], [7],
        #                         [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
        #                         [6, 7], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]


        null_idx = []
        # Load all data.
        for trial_idx, data_fn in enumerate(tqdm.tqdm(data_fns)):
            example_dict = mmint_utils.load_gzip_pickle(os.path.join(dataset_dir, data_fn))
            try:
                self.contact_patch.append(example_dict["test"]["contact_patch"])
            except:
                self.contact_patch.append(
                    self.surface_points[-1][self.surface_in_contact[-1]]
                )

            if len(self.contact_patch[-1]) == 0:
                null_idx.append(trial_idx)
                self.contact_patch.pop(-1)
                continue


            # Populate example info.
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
            self.partial_pointcloud.append(example_dict["input"]['pointclouds'])
            self.points_iou.append(example_dict["test"]["points_iou"])
            self.occ_tgt.append(example_dict["test"]["occ_tgt"])

        # Load nominal geometry info.
        self.nominal_query_points = []
        self.nominal_sdf = []

        for object_idx, nominal_fn in enumerate(nominal_fns):
            example_dict_nom = mmint_utils.load_gzip_pickle(os.path.join(dataset_dir, nominal_fn))

            # Populate nominal info.
            self.nominal_query_points.append(example_dict_nom["query_points"])
            self.nominal_sdf.append(example_dict_nom["sdf"])

        [data_fns.pop(null_idx_) for null_idx_ in null_idx]
        self.num_trials = len(data_fns)
        # assert len(self.nominal_query_points) == max(self.object_idcs) + 1

    def _from_idx_to_pcd(self, partial_index, partial_pcds):

        partial_pcd_idxs = self.partial_pcd_idx[partial_index]
        combined_pcd = []
        for partial_pcd_idx_i in partial_pcd_idxs:
            if partial_pcds[partial_pcd_idx_i]['pointcloud'] is not None:
                pcd_i = partial_pcds[partial_pcd_idx_i]['pointcloud']
                combined_pcd.append(pcd_i)
        return combined_pcd

    def get_num_objects(self):
        return self.num_objects

    def get_num_trials(self):
        return self.num_trials

    def get_example_mesh(self, example_idx):
        mesh_fn = os.path.join(self.dataset_dir, "out_%d_mesh.obj" % example_idx)
        mesh = trimesh.load(mesh_fn)
        return mesh

    def __len__(self):
        return self.num_trials

    def __getitem__(self, index):
        object_index = 0
        partial_index = np.random.randint(0, len(self.partial_pcd_idx), size=1)[0]
        combined_pcd = self._from_idx_to_pcd(partial_index , self.partial_pointcloud[index])

        # When selected indexes are all bad, try two more.
        if len(combined_pcd) == 0:
            combined_pcd = self._from_idx_to_pcd(partial_index + 1 , self.partial_pointcloud[index])
        if len(combined_pcd) == 0:
            combined_pcd = self._from_idx_to_pcd(partial_index + 2 , self.partial_pointcloud[index])

        partial_pointcloud = np.concatenate(combined_pcd, axis=0)


        data_dict = {
            "object_idx": torch.tensor([object_index], device=self.device),
            "trial_idx": torch.tensor([index], device=self.device),
            "query_point": torch.tensor(self.query_points[index], dtype=self.dtype, device=self.device),
            "sdf": torch.from_numpy(self.sdf[index]).to(self.device),
            "normals": torch.tensor(self.normals[index], dtype=self.dtype, device=self.device),
            "in_contact": torch.from_numpy(self.in_contact[index].astype(int)).to(self.device),
            "pressure": torch.tensor([self.trial_pressure[index]], device=self.device),
            "wrist_wrench": torch.tensor(self.wrist_wrench[index], device=self.device).float(),
            "nominal_query_point": torch.tensor(self.nominal_query_points[object_index], dtype=self.dtype, device=self.device),
            "nominal_sdf": torch.tensor(self.nominal_sdf[object_index], device=self.device),
            "surface_points": torch.tensor(self.surface_points[index], dtype=self.dtype, device=self.device),
            "surface_in_contact": torch.tensor(self.surface_in_contact[index], device=self.device),
            "partial_pointcloud": torch.tensor(partial_pointcloud, dtype=self.dtype, device=self.device),
            "contact_patch": self.contact_patch[index],
            "points_iou": torch.tensor(self.points_iou[index], dtype=self.dtype, device=self.device),
            "occ_tgt": torch.tensor(self.occ_tgt[index], device=self.device),
        }

        if self.transform is not None:
            data_dict = self.transform(data_dict)

        return data_dict
