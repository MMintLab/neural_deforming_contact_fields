import torch
import torch.nn as nn


class BaseGenerator(object):
    """
    Base generator class.

    Generator is responsible for implementing an API generating the shared representations from
    various model classes. Not all representations need to necessarily be implemented.
    """

    def __init__(self, cfg: dict, model: nn.Module, device: torch.device = None):
        self.cfg = cfg
        self.model = model
        self.device = device

        self.generates_nominal_mesh = False
        self.generates_mesh = False
        self.generates_pointcloud = False
        self.generates_contact_patch = False
        self.generates_contact_labels = False
        self.generates_iou_labels = False

    def generate_nominal_mesh(self, data, metadata):
        raise NotImplementedError()

    def generate_latent(self, data):
        raise NotImplementedError()

    def generate_mesh(self, data, metadata):
        raise NotImplementedError()

    def generate_pointcloud(self, data, metadata):
        raise NotImplementedError()

    def generate_contact_patch(self, data, metadata):
        raise NotImplementedError()

    def generate_contact_labels(self, data, metadata):
        raise NotImplementedError()

    def generate_iou_labels(self, data, metadata):
        raise NotImplementedError()
