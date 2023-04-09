import numpy as np

from neural_contact_fields.data.tool_dataset import ToolDataset


class NoiseWrenchToolDataset(ToolDataset):

    def __init__(self, dataset_dir, load_data=True, transform=None, device="cpu"):
        super().__init__(dataset_dir, load_data, transform, device)

        # Calculate average wrench value along each dim.
        self.wrist_wrench = np.array(self.wrist_wrench)
        self.wrist_wrench_mean = np.mean(self.wrist_wrench, axis=0)
