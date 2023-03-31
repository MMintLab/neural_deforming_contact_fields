import numpy as np
import torch

from neural_contact_fields.data.tool_dataset import ToolDataset
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf


class ToolRotateDataset(ToolDataset):

    def get_num_trials(self):
        return self.num_trials * 4

    def __len__(self):
        return self.num_trials * 4

    def __getitem__(self, index):
        trial_index = torch.div(index, 4, rounding_mode="floor")
        rotation_index = index % 4  # Which of the 4 rotations to use.

        # Build transform around z vector based on rotation index.
        rotation = np.pi / 2.0 * rotation_index
        transform = pk.Transform3d(rot=tf.euler_angles_to_matrix(torch.tensor([0.0, 0.0, rotation]), "XYZ"),
                                   dtype=self.dtype)

        # Transform wrench by individual rotation force/torque.
        wrench = torch.tensor(self.wrist_wrench[trial_index]).unsqueeze(0).float()
        wrench[..., :3] = transform.transform_normals(wrench[..., :3])
        wrench[..., 3:] = transform.transform_normals(wrench[..., 3:])
        wrench = wrench.squeeze(0).cpu().numpy()

        object_index = self.object_idcs[trial_index]
        data_dict = {
            "object_idx": np.array([object_index]),
            "trial_idx": np.array([index.cpu().numpy()]),
            "query_point": transform.transform_points(
                torch.tensor(self.query_points[trial_index], dtype=self.dtype)).cpu().numpy(),
            "sdf": self.sdf[trial_index],
            "normals": transform.transform_normals(
                torch.tensor(self.normals[trial_index], dtype=self.dtype)).cpu().numpy(),
            "in_contact": self.in_contact[trial_index].astype(int),
            "pressure": np.array([self.trial_pressure[trial_index]]),
            "wrist_wrench": wrench,
            "nominal_query_point": transform.transform_points(
                torch.tensor(self.nominal_query_points[object_index], dtype=self.dtype)).cpu().numpy(),
            "nominal_sdf": self.nominal_sdf[object_index],
            "surface_points": transform.transform_points(
                torch.tensor(self.surface_points[trial_index], dtype=self.dtype)).cpu().numpy(),
            "surface_in_contact": self.surface_in_contact[trial_index],
            "partial_pointcloud": transform.transform_points(
                torch.tensor(self.partial_pointcloud[trial_index], dtype=self.dtype)).cpu().numpy(),
            "contact_patch": transform.transform_points(
                torch.tensor(self.contact_patch[trial_index], dtype=self.dtype)).cpu().numpy(),
        }

        return data_dict
