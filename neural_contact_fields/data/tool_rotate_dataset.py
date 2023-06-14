import pdb

import numpy as np
import torch

from neural_contact_fields.data.tool_dataset import ToolDataset
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf


class ToolRotateDataset(ToolDataset):

    def get_num_trials(self):
        return self.original_num_trials * 4

    def get_example_mesh(self, example_idx):
        trial_index = torch.div(example_idx, 4, rounding_mode="floor")
        rotation_index = example_idx % 4  # Which of the 4 rotations to use.
        mesh = super().get_example_mesh(trial_index)

        # Build transform around z vector based on rotation index.
        rotation = np.pi / 2.0 * rotation_index
        transform = pk.Transform3d(rot=tf.euler_angles_to_matrix(torch.tensor([0.0, 0.0, rotation]), "XYZ"),
                                   dtype=self.dtype)

        mesh.apply_transform(transform.get_matrix().cpu().numpy()[0])
        return mesh

    def __len__(self):
        return self.num_trials * 4

    def __getitem__(self, index):
        dataset_index = torch.div(index, 4, rounding_mode="floor")
        trial_index = self.trial_idcs[dataset_index]
        rotation_index = index % 4  # Which of the 4 rotations to use.
        env_class = trial_index // (self.original_num_trials // 3)

        # Build transform around z vector based on rotation index.
        rotation = np.pi / 2.0 * rotation_index
        transform = pk.Transform3d(
            rot=tf.euler_angles_to_matrix(torch.tensor([0.0, 0.0, rotation], device=self.device), "XYZ"),
            dtype=self.dtype, device=self.device)

        # Transform wrench by individual rotation force/torque.
        wrench = torch.tensor(self.wrist_wrench[dataset_index], device=self.device).unsqueeze(0).float()
        wrench[..., :3] = transform.transform_normals(wrench[..., :3])
        wrench[..., 3:] = transform.transform_normals(wrench[..., 3:])
        wrench = wrench.squeeze(0)

        # Get partial pointcloud.
        partial_pointcloud = self.get_partial_pointcloud(dataset_index)

        object_index = self.object_idcs[dataset_index]
        data_dict = {
            "env_class": env_class,
            "object_idx": torch.tensor([object_index], device=self.device),
            "trial_idx": torch.tensor([trial_index * 4 + rotation_index], device=self.device),
            "query_point": transform.transform_points(
                torch.tensor(self.query_points[dataset_index], dtype=self.dtype, device=self.device)),
            "sdf": torch.from_numpy(self.sdf[dataset_index]).to(self.device),
            "normals": transform.transform_normals(
                torch.tensor(self.normals[dataset_index], dtype=self.dtype, device=self.device)),
            "in_contact": torch.from_numpy(self.in_contact[dataset_index].astype(int)).to(self.device),
            "pressure": torch.tensor([self.trial_pressure[dataset_index]], device=self.device),
            "wrist_wrench": wrench,
            "nominal_query_point": transform.transform_points(
                torch.tensor(self.nominal_query_points[object_index], dtype=self.dtype, device=self.device)),
            "nominal_sdf": torch.tensor(self.nominal_sdf[object_index], device=self.device),
            "surface_points": transform.transform_points(
                torch.tensor(self.surface_points[dataset_index], dtype=self.dtype, device=self.device)),
            "surface_in_contact": torch.tensor(self.surface_in_contact[dataset_index], device=self.device),
            "partial_pointcloud": transform.transform_points(
                torch.tensor(partial_pointcloud, dtype=self.dtype, device=self.device)),
            "contact_patch": transform.transform_points(
                torch.tensor(self.contact_patch[dataset_index], dtype=self.dtype, device=self.device)),
            "points_iou": transform.transform_points(
                torch.tensor(self.points_iou[dataset_index], dtype=self.dtype, device=self.device)),
            "occ_tgt": torch.tensor(self.occ_tgt[dataset_index], device=self.device),
        }

        if self.transform is not None:
            data_dict = self.transform(data_dict)

        return data_dict
