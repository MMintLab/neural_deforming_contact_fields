import numpy


class NoiseTransform(object):
    def __init__(self, noise_level):
        self.noise_level = noise_level

    def __call__(self, data_dict):
        # Create copy of partial pointcloud to avoid modifying the original.
        partial_pointcloud = data_dict["partial_pointcloud"].copy()

        # Add noise to points.
        noise = numpy.random.normal(0, self.noise_level, partial_pointcloud.shape)
        partial_pointcloud += noise

        # Update points in data dict.
        data_dict["partial_pointcloud"] = partial_pointcloud
        return data_dict
