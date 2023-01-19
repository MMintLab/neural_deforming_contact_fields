import torch
import torch.nn as nn


class BaseVisualizer(object):
    """
    Base visualizer class.

    Use for visualizing additional model outputs (beyond the default generated values).
    """

    def __init__(self, cfg: dict, model: nn.Module, device: torch.device = None, visualizer_args: dict = None):
        self.cfg = cfg
        self.model = model
        self.device = device
        self.visualizer_args = visualizer_args
        if self.visualizer_args is None:
            self.visualizer_args = dict()

    def visualize_pretrain(self, data: dict):
        raise NotImplementedError()

    def visualize_train(self, data: dict):
        raise NotImplementedError()

    def visualize_test(self, data: dict):
        raise NotImplementedError()
