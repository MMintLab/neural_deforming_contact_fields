import torch.nn as nn
import torch
from neural_contact_fields.models.mlp import build_mlp


class SingleToolNeuralContactField(nn.Module):

    def __init__(self, num_trials: int, z: int = 64, forward_deformation: bool = False, device=None):
        super().__init__()
        self.device = device
        self.forward_deformation = forward_deformation
        self.num_trials = num_trials
        self.z = z

        # Trial embedding.
        self.trial_embedding = nn.Embedding(self.num_trials, self.z).requires_grad_(True).to(self.device)

        # Object module - query point -> SDF on undeformed surface.
        self.object_module = build_mlp(3, 1, [64, 64, 32], device=self.device)

        # Deformation module - query point x trial code -> change in query point for trial.
        self.deformation_module = build_mlp(self.z + 3, 3, [128, 64, 32], device=self.device)

        # Contact module - query point x trial code [x def at p] -> binary contact x force.
        self.contact_module = build_mlp(self.z + 3 + (3 if self.forward_deformation else 0), 4, [128, 64, 32],
                                        device=self.device)

    def forward_object_module(self, query_points: torch.Tensor):
        return self.object_module.forward(query_points).squeeze(-1)

    def forward(self, trial_idcs: torch.Tensor, coords: torch.Tensor):
        # Embed trials.
        z = self.trial_embedding.forward(trial_idcs)

        # Find deformation at each query point.
        def_in = torch.cat([z, coords], dim=1)
        delta_coords = self.deformation_module.forward(def_in)

        # Get deformed query point.
        def_coords = coords + delta_coords
        sdf = self.forward_object_module(def_coords)

        # Get contact prob/force.
        if self.forward_deformation:
            cm_in = torch.cat([z, coords, delta_coords], dim=1)
        else:
            cm_in = torch.cat([z, coords], dim=1)
        cm_out = self.contact_module(cm_in)

        # Get contact probability at each point.
        contact_logits = cm_out[:, 0]
        contact_prob = torch.sigmoid(contact_logits)

        # Get contact force at each point.
        contact_force = cm_out[:, 1:]

        return sdf, contact_logits, contact_prob, contact_force
