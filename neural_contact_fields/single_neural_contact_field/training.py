from collections import defaultdict
import numpy as np
import torch
from neural_contact_fields.training import BaseTrainer
from neural_contact_fields.loss import sdf_loss_clamp
import torch.nn.functional as F
from tqdm import tqdm


class Trainer(BaseTrainer):

    def __init__(self, model, optimizer, logger, loss_weights, vis_dir, device=None):
        """
        Args:
        - model (nn.Module): model
        - optimizer (optimizer): pytorch optimizer
        - logger (tensorboardX.SummaryWriter): logger for tensorboard
        - vis_dir (str): vis directory
        - device (device): pytorch device
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.loss_weights = loss_weights
        self.vis_dir = vis_dir

    def train_step(self, data, it):
        """
        Perform training step.

        Args:
        - data (dict): batch data dict
        - it (int): training iter
        """
        self.model.train()
        self.optimizer.zero_grad()
        loss_dict, _ = self.compute_loss(data, it)

        for k, v in loss_dict.items():
            self.logger.add_scalar(k, v, it)

        loss = loss_dict['loss']
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validation(self, val_loader, it):
        val_dict = defaultdict(list)

        for val_batch in tqdm(val_loader):
            eval_loss_dict, eval_out_dict = self.eval_step(val_batch, it)

            val_dict["loss"].append(eval_loss_dict["loss"].item())

        return {
            "val_loss": np.mean(val_dict["loss"]),
        }

    def eval_step(self, data, it):
        """
        Perform evaluation step.

        Args:
        - data (dict): data dictionary
        """
        self.model.eval()

        with torch.no_grad():
            loss_dict, out_dict = self.compute_loss(data, it)

        return loss_dict, out_dict

    def visualize(self, data):
        """
        Visualize the predicted RGB/Depth images.
        """
        # TODO: Visualize?
        pass

    def compute_loss(self, data, it):
        """
        Get loss over provided batch of data.

        Args:
        - data (dict): data dictionary
        """
        device = self.device
        loss_dict = dict()
        out_dict = dict()

        # Pull out data.
        coord = data["coord"].to(device)
        sdf_gt = data["sdf"].to(device)
        contact_gt = data["contact"].to(device)

        # Map to SDF/Contact.
        sdf_pred, contact_logits_pred, contact_pred = self.model(coord)

        # Losses.
        sdf_loss = sdf_loss_clamp(sdf_pred, sdf_gt, clamp=0.3, reduce="mean")

        contact_loss = F.binary_cross_entropy_with_logits(contact_logits_pred, contact_gt, reduction="mean")

        loss = (10 * sdf_loss) + contact_loss  # TODO: Add weighting.

        # Construct out dicts.
        loss_dict["sdf"] = sdf_loss
        loss_dict["contact"] = contact_loss
        loss_dict["loss"] = loss

        out_dict["sdf"] = sdf_pred
        out_dict["contact_logits"] = contact_logits_pred
        out_dict["contact"] = contact_pred

        return loss_dict, out_dict
