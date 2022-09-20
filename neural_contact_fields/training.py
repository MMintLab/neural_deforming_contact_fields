from collections import defaultdict

import numpy as np
from tqdm import tqdm


class BaseTrainer(object):
    """
    Base trainer class.

    Copied from: https://github.com/autonomousvision/occupancy_networks/blob/master/im2mesh/training.py
    """

    def evaluate(self, val_loader):
        """
        Performs an evaluation.

        Args:
            val_loader (dataloader): pytorch dataloader
        """
        eval_list = defaultdict(list)

        for data in tqdm(val_loader):
            eval_step_dict = self.eval_step(data)

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        return eval_dict

    def validation(self, val_loader, it):
        """
        Performs validation on the given dataset.
        Should return a dictionary with key "val_loss" including
        the metric to select best model with.
        """
        raise NotImplementedError

    def train_step(self, *args, **kwargs):
        """
        Performs a training step.
        """
        raise NotImplementedError

    def eval_step(self, *args, **kwargs):
        """
        Performs an evaluation step.
        """
        raise NotImplementedError

    def visualize(self, *args, **kwargs):
        """
        Performs  visualization.
        """
        raise NotImplementedError
