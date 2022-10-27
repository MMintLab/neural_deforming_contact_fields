import torch.nn as nn
import torch


class SingleToolNeuralContactField(nn.Module):

    def __init__(self, device=None):
        super().__init__()

        self.device = device

    def forward(self, coords: torch.Tensor):
        raise NotImplementedError()
