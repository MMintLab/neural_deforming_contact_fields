import torch.nn as nn
import torch


class NeuralContactField(nn.Module):
    """
    Neural Contact Field abstract class.
    """

    def forward_object_module(self, query_points: torch.Tensor, z_object: torch.Tensor):
        raise NotImplementedError()

    def object_module_regularization_loss(self, out_dict: dict):
        raise NotImplementedError()

    def forward(self, query_points: torch.Tensor, z_deform: torch.Tensor, z_object: torch.Tensor):
        raise NotImplementedError()

    def regularization_loss(self, out_dict: dict):
        raise NotImplementedError()
