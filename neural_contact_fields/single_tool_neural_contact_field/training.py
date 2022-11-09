import pdb
from collections import defaultdict
import numpy as np
import torch
from neural_contact_fields.training import BaseTrainer
from neural_contact_fields.loss import sdf_loss_clamp
from tqdm import tqdm
import torch.nn.functional as F


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

        # Send data to torch/device.
        trial_idx = data["trial_idx"].long().to(device)
        query_point = data["query_point"].float().to(device)
        sdf = data["sdf"].float().to(device)
        in_contact = data["in_contact"].to(device)
        in_contact_float = torch.clone(in_contact).float()
        force = data["force"].float().to(device)

        pred_dict = self.model.forward(trial_idx, query_point)

        # Apply losses.

        # Apply L2 loss on trial embeddings.
        embed_loss = torch.linalg.norm(pred_dict["z"]).mean()
        loss_dict["embed_loss"] = embed_loss

        # We apply the SDF loss to every point in space.
        sdf_loss = F.l1_loss(pred_dict["sdf"], sdf, reduction="mean")
        loss_dict["sdf_loss"] = sdf_loss

        # Apply L2 loss on deformation.
        def_loss = torch.linalg.norm(pred_dict["delta_coords"]).mean()
        loss_dict["def_loss"] = def_loss

        # Next, for all points *on the surface* we predict the contact probability.
        surface_query_points = sdf == 0.0
        if surface_query_points.sum() > 0:
            contact_loss = F.binary_cross_entropy_with_logits(pred_dict["contact_logits"][surface_query_points],
                                                              in_contact_float[surface_query_points], reduction="mean")
            loss_dict["contact_loss"] = contact_loss
        else:
            loss_dict["contact_loss"] = 0.0

        # Finally, for all points *in contact* we predict the contact forces.
        if in_contact.sum() > 0:
            force_loss = F.mse_loss(pred_dict["contact_force"][in_contact], force[in_contact], reduction="mean")
            loss_dict["force_loss"] = force_loss
        else:
            loss_dict["force_loss"] = 0.0

        # Combined losses.
        loss = (self.loss_weights["sdf_loss"] * sdf_loss) + \
               (self.loss_weights["def_loss"] * def_loss) + \
               (self.loss_weights["contact_loss"] * loss_dict["contact_loss"]) + \
               (self.loss_weights["force_loss"] * loss_dict["force_loss"])
        loss_dict["loss"] = loss

        return loss_dict, out_dict
