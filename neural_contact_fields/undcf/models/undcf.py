import torch
from torch.distributions import Normal

from neural_contact_fields.models import meta_modules, mlp, point_net
import neural_contact_fields.loss as ncf_losses
from neural_contact_fields.utils import diff_operators
from torch import nn


class UNDCF(nn.Module):
    """
    Neural Contact Field using Virdo sub-modules.
    """

    def __init__(self, z_deform_size: int, z_wrench_size: int, device=None):
        super().__init__()
        self.device = device
        self.z_deform_size = z_deform_size
        self.z_wrench_size = z_wrench_size

        self.no_wrench = self.z_wrench_size == 0
        self.no_trial_code = self.z_deform_size == 0

        # Setup sub-models of the VirdoNCF.
        combined_latent_size = self.z_deform_size + self.z_wrench_size
        self.contact_model = meta_modules.virdo_hypernet(
            in_features=3, out_features=2, hyper_in_features=combined_latent_size, hl=2
        ).to(self.device)

        self.object_model = meta_modules.virdo_hypernet(
            in_features=3, out_features=2, hyper_in_features=combined_latent_size, hl=2
        ).to(self.device)

        self.wrench_encoder = mlp.build_mlp(6, self.z_wrench_size, hidden_sizes=[16], device=device)

        # Pointcloud encoder!
        self.pointcloud_encoder = point_net.PointNet(self.z_deform_size, pooling="max").to(self.device)

    def encode_wrench(self, wrench: torch.Tensor):
        z_wrench = self.wrench_encoder(wrench)
        return z_wrench

    def encode_pointcloud(self, pointcloud: torch.Tensor):
        z_deform = self.pointcloud_encoder(pointcloud)
        return z_deform

    def forward(self, query_points: torch.Tensor, z_deform: torch.Tensor, z_wrench: torch.Tensor):
        combined_embedding = torch.cat([z_deform, z_wrench], dim=-1)
        model_in = {
            "coords": query_points,
            "embedding": combined_embedding,
        }

        # Get SDF distribution at each query point.
        sdf_out = self.object_model(model_in)
        sdf_means = sdf_out["model_out"][..., 0]
        sdf_var = torch.exp(sdf_out["model_out"][..., 1]) + 1e-10  # Ensure positive.
        sdf_dist = Normal(sdf_means, sdf_var)

        # Calculate normals w.r.t. to distribution mean.
        pred_normals = diff_operators.gradient(sdf_means, sdf_out["model_in"])

        # Get contact label at each query point.
        contact_out = self.contact_model(model_in)
        in_contact_logit_means = contact_out["model_out"][..., 0]
        in_contact_logit_var = torch.exp(contact_out["model_out"][..., 1]) + 1e-10  # Ensure positive.
        in_contact_logit_dist = Normal(in_contact_logit_means, in_contact_logit_var)
        in_contact = torch.sigmoid(in_contact_logit_means)

        out_dict = {
            "query_points": sdf_out["model_in"],
            "sdf_dist": sdf_dist,
            "sdf_hypo_params": sdf_out["hypo_params"],
            "in_contact_dist": in_contact_logit_dist,
            "in_contact": in_contact,
            "in_contact_hypo_params": contact_out["hypo_params"],
            "embedding": combined_embedding,
            "normals": pred_normals,
        }
        return out_dict

    def regularization_loss(self, out_dict: dict):
        sdf_hypo_params = out_dict["sdf_hypo_params"]
        sdf_hypo_loss = ncf_losses.hypo_weight_loss(sdf_hypo_params)
        in_contact_hypo_params = out_dict["in_contact_hypo_params"]
        in_contact_hypo_loss = ncf_losses.hypo_weight_loss(in_contact_hypo_params)
        return sdf_hypo_loss + in_contact_hypo_loss
