import torch.nn as nn
import torch
from neural_contact_fields.models.mlp import build_mlp


class SingleNeuralContactField(nn.Module):

    def __init__(self, device=None):
        super().__init__()

        self.device = device

        self.mlp = build_mlp(3, 64, [128, 128, 128, 64, 64], device=device)
        self.sdf_head = build_mlp(64, 1, [], device=device)
        self.contact_head = build_mlp(64, 1, [], device=device)

    def forward(self, coords: torch.Tensor):
        z = self.mlp(coords)
        sdf = torch.tanh(self.sdf_head(z))
        contact_logits = self.contact_head(z)
        contact = torch.sigmoid(contact_logits)

        return sdf, contact_logits, contact
