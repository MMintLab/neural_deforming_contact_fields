import numpy as np

from neural_contact_fields.data.tool_dataset import ToolDataset
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf


class ToolRotateDataset(ToolDataset):

    def get_num_trials(self):
        return self.num_trials * 4

    def __len__(self):
        return self.num_trials * 4

    def __getitem__(self, index):
        trial_index = index // 4
        rotation_index = index % 4  # Which of the 4 rotations to use.

        # Build transform around z vector based on rotation index.
        rotation = np.pi / 2.0 * rotation_index
        transform = pk.Transform3d(rot=tf.euler_angles_to_matrix([0.0, 0.0, rotation], "XYZ"))

        # Transform wrench by individual rotation force/torque.
        wrench = self.wrist_wrench[trial_index]
        wrench[..., :3] = transform.transform_normals(wrench[..., :3])
        wrench[..., 3:] = transform.transform_normals(wrench[..., 3:])

        object_index = self.object_idcs[trial_index]
        data_dict = {
            "object_idx": np.array([object_index]),
            "trial_idx": np.array([index]),
            "query_point": transform.transform_points(self.query_points[trial_index]),
            "sdf": self.sdf[trial_index],
            "normals": transform.transform_normals(self.normals[trial_index]),
            "in_contact": self.in_contact[trial_index].astype(int),
            "pressure": np.array([self.trial_pressure[trial_index]]),
            "wrist_wrench": wrench,
            "nominal_query_point": transform.transform_points(self.nominal_query_points[object_index]),
            "nominal_sdf": self.nominal_sdf[object_index],
            "surface_points": transform.transform_points(self.surface_points[trial_index]),
            "surface_in_contact": self.surface_in_contact[trial_index],
            "partial_pointcloud": transform.transform_points(self.partial_pointcloud[trial_index]),
            "contact_patch": transform.transform_points(self.contact_patch[trial_index]),
        }

        return data_dict
