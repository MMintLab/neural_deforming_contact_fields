import os

import torch.nn as nn
import torch
from neural_contact_fields.models import mlp
from neural_contact_fields.neural_contact_field.models.neural_contact_field import NeuralContactField
from neural_contact_fields.utils import diff_operators


class MLPNCF(NeuralContactField):

    def __init__(self, num_objects: int, num_trials: int, z_object_size: int, z_deform_size: int, z_wrench_size: int,
                 device=None):
        super().__init__(z_object_size, z_deform_size, z_wrench_size)
        self.device = device

        # Setup sub-models of the MLP-NCF.
        self.object_model = mlp.build_mlp(3 + self.z_object_size, 1, hidden_sizes=[64, 32], device=self.device)
        self.deformation_model = mlp.build_mlp(3 + self.z_object_size + self.z_deform_size + self.z_wrench_size, 3,
                                               hidden_sizes=[64, 32], device=self.device)
        self.contact_model = mlp.build_mlp(3 + self.z_object_size + self.z_deform_size + self.z_wrench_size, 1,
                                           hidden_sizes=[64, 32], device=self.device)
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
        query_points.requires_grad_(True)
        z_object_tiled = z_object.repeat(1, query_points.shape[1], 1)
        model_in = torch.cat([query_points, z_object_tiled], dim=-1)

        model_out = self.object_model(model_in)

        # Calculate normals.
        if normal_query_points is None:
            normal_query_points = query_points
        pred_normals = diff_operators.gradient(model_out.squeeze(-1), normal_query_points)

        out_dict = {
            "query_points": model_in,
            "sdf": model_out.squeeze(-1),
            "embedding": z_object,
            "normals": pred_normals,
        }
        return out_dict

    def object_module_regularization_loss(self, out_dict: dict):
        return 0.0

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
        query_points.requires_grad_(True)

        # Concatenate embeddings to query points.
        combined_embedding = torch.cat([z_deform, z_object, z_wrench], dim=-1)
        combined_embedding_tiled = combined_embedding.repeat(1, query_points.shape[1], 1)
        model_in = torch.cat([query_points, combined_embedding_tiled], dim=-1)

        # Predict deformation.
        deform_out = self.deformation_model(model_in)

        # Apply deformation to query points.
        object_coords = query_points + deform_out

        # Predict SDF.
        object_out = self.forward_object_module(object_coords, z_object, query_points)

        # Predict contact.
        in_contact_logits = self.contact_model(model_in).squeeze(-1)
        in_contact = torch.sigmoid(in_contact_logits)

        out_dict = {
            "query_points": query_points,
            "deform": deform_out,
            "nominal": object_coords,
            "sdf": object_out["sdf"],
            "in_contact_logits": in_contact_logits,
            "in_contact": in_contact,
            "embedding": combined_embedding,
            "normals": object_out["normals"],
        }
        return out_dict

    def regularization_loss(self, out_dict: dict):
        return 0.0
