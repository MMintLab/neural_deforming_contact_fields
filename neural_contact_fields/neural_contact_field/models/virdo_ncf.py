import os

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

    def __init__(self, num_objects: int, num_trials: int, z_object_size: int, z_deform_size: int, z_wrench_size: int,
                 forward_deformation_input: bool, device=None):
        super().__init__(z_object_size, z_deform_size, z_wrench_size)
        self.device = device
        self.forward_deformation_input = forward_deformation_input

        self.no_wrench = self.z_wrench_size == 0

        # Setup sub-models of the VirdoNCF.
        self.object_model = meta_modules.virdo_hypernet(in_features=3, out_features=1,
                                                        hyper_in_features=self.z_object_size, hl=2).to(self.device)
        self.deformation_model = meta_modules.virdo_hypernet(
            in_features=3, out_features=3,
            hyper_in_features=self.z_object_size + self.z_deform_size + self.z_wrench_size, hl=1
        ).to(self.device)

        combined_latent_size = self.z_object_size + self.z_deform_size + self.z_wrench_size
        self.contact_model = meta_modules.virdo_hypernet(
            in_features=6 if self.forward_deformation_input else 3,
            out_features=1,
            hyper_in_features=combined_latent_size,
            hl=2
        ).to(self.device)

        self.wrench_encoder = mlp.build_mlp(6, self.z_wrench_size, hidden_sizes=[16], device=device)

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

    def load_pretrained_model(self, pretrain_file: str, load_pretrain_cfg: dict):
        # Load object model from pretrain state dict.
        if os.path.exists(pretrain_file):
            print('Loading checkpoint from local file: %s' % pretrain_file)
            state_dict = torch.load(pretrain_file, map_location='cpu')
            pretrain_object_module_dict = {k.replace("object_model.", ""): state_dict["model"][k] for k in
                                           state_dict["model"].keys() if "object_model" in k}

            missing_keys, _ = self.object_model.load_state_dict(pretrain_object_module_dict)
            if len(missing_keys) != 0:
                raise Exception("Pretrain model missing keys!")
        else:
            raise Exception("Couldn't find pretrain file: %s" % pretrain_file)

        # Freeze object module, if requested.
        freeze_object_module_weights = load_pretrain_cfg["freeze_object_module_weights"]
        if freeze_object_module_weights:
            for param in self.object_model.parameters():
                param.requires_grad = False

    def encode_trial(self, object_idx: torch.Tensor, trial_idx: torch.Tensor):
        z_object = self.encode_object(object_idx)
        z_trial = self.trial_code(trial_idx)
        return z_object, z_trial

    def encode_wrench(self, wrench: torch.Tensor):
        z_wrench = self.wrench_encoder(wrench)
        return z_wrench

    def forward(self, query_points: torch.Tensor, z_deform: torch.Tensor, z_object: torch.Tensor,
                z_wrench: torch.Tensor):
        if self.no_wrench:
            combined_embedding = torch.cat([z_deform, z_object], dim=-1)
        else:
            combined_embedding = torch.cat([z_deform, z_object, z_wrench], dim=-1)

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
        combined_query_points_def = torch.cat([query_points, query_point_defs], dim=-1)
        contact_in = {
            "coords": combined_query_points_def if self.forward_deformation_input else query_points,
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
