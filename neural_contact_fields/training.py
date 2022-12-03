import os

from neural_contact_fields.utils import model_utils
from torch.utils.data import Dataset


class BaseTrainer(object):
    """
    Base trainer class.
    """

    def __init__(self, cfg, model, device=None):
        """
        Args:
        - cfg (dict): trainer configuration
        - model (nn.Module): model
        - device (device): pytorch device
        """
        self.cfg = cfg
        self.model = model
        self.device = device

    def load_partial_train_model(self, optimizer, out_dir, partial_model_fn):
        """
        Loads a partially trained model from the given out/fn and attempts to load optimizer state.

        Args:
        - optimizer (torch.optim.Optimizer): optimizer being used in training
        - out_dir (str): out directory where files are being saved
        - partial_model_fn (str): file where partially trained model is
        """
        model_dict = {
            'model': self.model,
            'optimizer': optimizer,
        }
        model_file = os.path.join(out_dir, partial_model_fn)
        load_dict = model_utils.load_model(model_dict, model_file)
        epoch_it = load_dict.get('epoch_it', -1)
        it = load_dict.get('it', -1)
        return epoch_it, it

    def train_step(self, data, it, optimizer, logger, compute_loss_fn):
        """
        Perform training step. This wraps up the gradient calculation for convenience.

        Args:
        - data (dict): batch data dict
        - it (int): training iter
        - optimizer (torch.optim.Optimizer): optimizer used for training
        - logger: tensorboard logger being used.
        - compute_loss_fn (callable): loss function to call. Should return a loss dictionary with main
          loss at key "loss" and an out_dict
        """
        self.model.train()
        optimizer.zero_grad()
        loss_dict, _ = compute_loss_fn(data, it)

        for k, v in loss_dict.items():
            logger.add_scalar(k, v, it)

        loss = loss_dict['loss']
        loss.backward()
        optimizer.step()

        return loss.item()

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
