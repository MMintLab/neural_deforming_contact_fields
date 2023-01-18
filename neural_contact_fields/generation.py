import torch
import torch.nn as nn
import trimesh


class BaseGenerator(object):

    def __init__(self, cfg: dict, model: nn.Module, device: torch.device = None):
        self.cfg = cfg
        self.model = model
        self.device = device

        self.generates_mesh = False
        self.generates_pointcloud = False
        self.generates_contact_patch = False
        self.generates_contact_labels = False

    def generate_mesh(self, data, metadata):
        raise NotImplementedError()

    def generate_pointcloud(self, data, metadata):
        raise NotImplementedError()

    def generate_contact_patch(self, data, metadata):
        raise NotImplementedError()

    def generate_contact_labels(self, data, metadata):
        raise NotImplementedError()
