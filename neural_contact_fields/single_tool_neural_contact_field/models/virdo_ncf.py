import torch
from neural_contact_field import NeuralContactField


class VirdoNCF(NeuralContactField):

    def __init__(self):
        super().__init__()

        self.object_model = None
        self.deformation_model = None
        self.contact_model = None

    def forward_object_module(self, query_points: torch.Tensor, z_object: torch.Tensor):
        pass

    def forward(self, query_points: torch.Tensor, z_deform: torch.Tensor, z_object: torch.Tensor):
        pass

    def regularization_loss(self):
        pass
