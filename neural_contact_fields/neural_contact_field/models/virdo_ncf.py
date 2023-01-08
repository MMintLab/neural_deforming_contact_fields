import torch
from neural_contact_fields.neural_contact_field.models.neural_contact_field import NeuralContactField
from neural_contact_fields.models import meta_modules, mlp
import neural_contact_fields.loss as ncf_losses
from neural_contact_fields.utils import diff_operators
from torch import nn


class VirdoNCF(NeuralContactField):
    """
    Neural Contact Field using Virdo sub-modules.
    """

    def __init__(self, num_objects: int, num_trials: int, z_object_size: int, z_deform_size: int, z_pressure_size: int,
                 device=None):
        super().__init__(z_object_size, z_deform_size, z_pressure_size)
        self.device = device

        # Setup sub-models of the VirdoNCF.
        self.object_model = meta_modules.virdo_hypernet(in_features=3, out_features=1,
                                                        hyper_in_features=self.z_object_size, hl=2).to(self.device)
        self.deformation_model = meta_modules.virdo_hypernet(
            in_features=3, out_features=3,
            hyper_in_features=self.z_object_size + self.z_deform_size + self.z_pressure_size, hl=1
        ).to(self.device)
        self.contact_model = meta_modules.virdo_hypernet(
            in_features=3, out_features=1,
            hyper_in_features=self.z_object_size + self.z_deform_size + self.z_pressure_size, hl=2
        ).to(self.device)
        self.pressure_encoder = mlp.build_mlp(1, z_pressure_size, hidden_sizes=None, device=device)

        # Setup latent embeddings (used during training).
        self.object_code = nn.Embedding(num_objects, self.z_object_size, dtype=torch.float32).requires_grad_(True).to(
            self.device)
        nn.init.normal_(self.object_code.weight, mean=0.0, std=0.1)
        self.trial_code = nn.Embedding(num_trials, self.z_deform_size, dtype=torch.float32).requires_grad_(True).to(
            self.device)
        nn.init.normal_(self.trial_code.weight, mean=0.0, std=0.1)

    def encode_object(self, object_idx: torch.Tensor):
        return self.object_code(object_idx)

    def forward_object_module(self, query_points: torch.Tensor, z_object: torch.Tensor,
                              normal_query_points: torch.Tensor = None):
        if normal_query_points is None:
            model_in = {
                "coords": query_points,
                "embedding": z_object,
            }
        else:
            model_in = {
                "coords": normal_query_points,
                "model_out": query_points,
                "embedding": z_object
            }

        model_out = self.object_model(model_in)

        # Calculate normals.
        if normal_query_points is None:
            normal_query_points = model_out["model_in"]
        pred_normals = diff_operators.gradient(model_out["model_out"].squeeze(-1), normal_query_points)

        out_dict = {
            "query_points": model_out["model_in"],
            "sdf": model_out["model_out"].squeeze(-1),
            "hypo_params": model_out["hypo_params"],
            "embedding": z_object,
            "normals": pred_normals,
        }
        return out_dict

    def object_module_regularization_loss(self, out_dict: dict):
        hypo_params = out_dict["hypo_params"]
        hypo_loss = ncf_losses.hypo_weight_loss(hypo_params)
        return hypo_loss

    def encode_trial(self, object_idx: torch.Tensor, trial_idx: torch.Tensor):
        z_object = self.encode_object(object_idx)
        z_trial = self.trial_code(trial_idx)
        return z_object, z_trial

    def encode_pressure(self, pressure: torch.Tensor):
        z_pressure = self.pressure_encoder(pressure)
        return z_pressure

    def forward(self, query_points: torch.Tensor, z_deform: torch.Tensor, z_object: torch.Tensor,
                z_pressure: torch.Tensor):
        combined_embedding = torch.cat([z_deform, z_object, z_pressure], dim=-1)

        # Determine deformation at each query point.
        deform_in = {
            "coords": query_points,
            "embedding": combined_embedding,
        }
        deform_out = self.deformation_model(deform_in)
        query_point_defs = deform_out["model_out"]

        # Apply deformation to query points.
        object_coords = deform_out["model_in"] + deform_out["model_out"]
        object_out = self.forward_object_module(object_coords, z_object, deform_out["model_in"])

        # Get contact label at each query point.
        contact_in = {
            "coords": query_points,
            "embedding": combined_embedding,
        }
        contact_out = self.contact_model(contact_in)
        in_contact_logits = contact_out["model_out"].squeeze(-1)
        in_contact = torch.sigmoid(in_contact_logits)

        out_dict = {
            "query_points": object_out["query_points"],
            "deform": query_point_defs,
            "nominal": object_coords,
            "def_hypo_params": deform_out["hypo_params"],
            "sdf": object_out["sdf"],
            "sdf_hypo_params": object_out["hypo_params"],
            "in_contact_logits": in_contact_logits,
            "in_contact": in_contact,
            "in_contact_hypo_params": contact_out["hypo_params"],
            "embedding": combined_embedding,
            "normals": object_out["normals"],
        }
        return out_dict

    def regularization_loss(self, out_dict: dict):
        def_hypo_params = out_dict["def_hypo_params"]
        def_hypo_loss = ncf_losses.hypo_weight_loss(def_hypo_params)
        sdf_hypo_params = out_dict["sdf_hypo_params"]
        sdf_hypo_loss = ncf_losses.hypo_weight_loss(sdf_hypo_params)
        in_contact_hypo_params = out_dict["in_contact_hypo_params"]
        in_contact_hypo_loss = ncf_losses.hypo_weight_loss(in_contact_hypo_params)
        return def_hypo_loss + sdf_hypo_loss + in_contact_hypo_loss
