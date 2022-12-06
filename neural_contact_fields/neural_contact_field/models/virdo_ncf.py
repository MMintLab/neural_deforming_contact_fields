import torch
from neural_contact_field import NeuralContactField
from neural_contact_fields.models import meta_modules


class VirdoNCF(NeuralContactField):
    """
    Neural Contact Field using Virdo sub-modules.
    """

    def __init__(self, z_object_size: int, z_deform_size: int, device=None):
        super().__init__()
        self.z_object_size = z_object_size
        self.z_deform_size = z_deform_size
        self.device = device

        # Setup sub-models of the VirdoNCF.
        self.object_model = meta_modules.virdo_hypernet(in_features=3, out_features=1,
                                                        hyper_in_features=self.z_object_size, hl=2).to(self.device)
        self.deformation_model = meta_modules.virdo_hypernet(in_features=3, out_features=3,
                                                             hyper_in_features=self.z_object_size + self.z_deform_size,
                                                             hl=1).to(self.device)
        self.contact_model = meta_modules.virdo_hypernet(in_features=3, out_features=1,
                                                         hyper_in_features=self.z_object_size + self.z_deform_size,
                                                         hl=2).to(self.device)

    def forward_object_module(self, query_points: torch.Tensor, z_object: torch.Tensor):
        model_in = {
            "coords": query_points,
            "embedding": z_object,
        }

        model_out = self.object_model(model_in)
        return model_out

    def forward(self, query_points: torch.Tensor, z_deform: torch.Tensor, z_object: torch.Tensor):
        combined_embedding = torch.cat([z_deform, z_object], dim=-1)

        # Determine deformation at each query point.
        deform_in = {
            "coords": query_points,
            "embedding": combined_embedding,
        }
        deform_out = self.deformation_model(deform_in)
        query_point_defs = deform_out["model_output"]

        # Apply deformation to query points.
        object_coords = query_points + query_point_defs
        object_out = self.forward_object_module(object_coords, z_object)

        # Get contact label at each query point.
        contact_in = {
            "coords": query_points,
            "embedding": combined_embedding,
        }
        contact_out = self.contact_model(contact_in)

    def regularization_loss(self):
        pass
