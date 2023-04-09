import numpy as np


class WrenchNoiseTransform(object):

    def __init__(self, noise_percent):
        self.noise_percent = noise_percent

    def __call__(self, data_dict):
        # Create copy of partial pointcloud to avoid modifying the original.
        wrist_wrench = data_dict["wrist_wrench"].copy()

        # Add noise to points.
        noise = np.random.normal(0, self.noise_percent * np.abs(wrist_wrench))
        wrist_wrench += noise

        # Update points in data dict.
        data_dict["wrist_wrench"] = wrist_wrench
        return data_dict
