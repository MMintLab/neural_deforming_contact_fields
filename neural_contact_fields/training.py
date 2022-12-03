from collections import defaultdict

import numpy as np
from tqdm import tqdm


class BaseTrainer(object):
    """
    Base trainer class.

    Modified from: https://github.com/autonomousvision/occupancy_networks/blob/master/im2mesh/training.py
    """

    def pretrain(self, *args, **kwargs):
        """
        Pretraining for model.
        """
        raise NotImplementedError()

    def train(self, *args, **kwargs):
        """
        Main training loop.
        """
        raise NotImplementedError()
