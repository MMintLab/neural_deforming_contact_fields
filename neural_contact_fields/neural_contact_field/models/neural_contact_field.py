import torch.nn as nn
import torch


class NeuralContactField(nn.Module):
    """
    Neural Contact Field abstract class.
    """

    def __init__(self, z_object_size: int, z_deform_size: int):
        super().__init__()
        self.z_object_size = z_object_size
        self.z_deform_size = z_deform_size

    def encode_object(self, object_idx: torch.Tensor):
        raise NotImplementedError()

    def forward_object_module(self, query_points: torch.Tensor, z_object: torch.Tensor,
                              normal_query_points: torch.Tensor = None):
        raise NotImplementedError()

    def object_module_regularization_loss(self, out_dict: dict):
        raise NotImplementedError()

    def encode_trial(self, object_idx: torch.Tensor, trial_idx: torch.Tensor):
        raise NotImplementedError()

    def forward(self, query_points: torch.Tensor, z_deform: torch.Tensor, z_object: torch.Tensor):
        raise NotImplementedError()

    def regularization_loss(self, out_dict: dict):
        raise NotImplementedError()
