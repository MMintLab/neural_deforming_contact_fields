import pdb

import mmint_utils
import numpy as np
import torch
from neural_contact_fields.data.tool_dataset import ToolDataset
from neural_contact_fields.undcf.models.undcf import UNDCF
from neural_contact_fields.training import BaseTrainer
import os
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import Dataset
import neural_contact_fields.loss as ncf_losses


class Trainer(BaseTrainer):

    def __init__(self, cfg, model: UNDCF, device):
        super().__init__(cfg, model, device)

        self.model: UNDCF = model

    ##########################################################################
    #  Pretraining loop                                                      #
    ##########################################################################

    def pretrain(self, pretrain_dataset: Dataset):
        pass

    ##########################################################################
    #  Main training loop                                                    #
    ##########################################################################

    def train(self, train_dataset: ToolDataset, validation_dataset: ToolDataset):
        # Shorthands:
        out_dir = self.cfg['training']['out_dir']
        lr = self.cfg['training']['learning_rate']
        max_epochs = self.cfg['training']['epochs']
        epochs_per_save = self.cfg['training']['epochs_per_save']
        self.train_loss_weights = self.cfg['training']['loss_weights']  # TODO: Better way to set this?

        # Output + vis directory
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        logger = SummaryWriter(os.path.join(out_dir, 'logs/train'))

        # Dump config to output directory.
        mmint_utils.dump_cfg(os.path.join(out_dir, 'config.yaml'), self.cfg)

        # Get optimizer (TODO: Parameterize?)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Load model + optimizer if a partially trained copy of it exists.
        epoch_it, it = self.load_partial_train_model(
            {"model": self.model, "optimizer": optimizer}, out_dir, "model.pt")

        # Training loop
        while True:
            epoch_it += 1

            if epoch_it > max_epochs:
                print("Backing up and stopping training. Reached max epochs.")
                save_dict = {
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch_it': epoch_it,
                    'it': it,
                }
                torch.save(save_dict, os.path.join(out_dir, 'model.pt'))
                break

            loss = None

            trial_idcs = np.arange(len(train_dataset))
            np.random.shuffle(trial_idcs)
            trial_idcs = torch.from_numpy(trial_idcs).to(self.device)

            for trial_idx in trial_idcs:
                it += 1

                # For this training, we use just a single example per run.
                batch = train_dataset[trial_idx]
                loss = self.train_step(batch, it, optimizer, logger, self.compute_train_loss)

            print('[Epoch %02d] it=%03d, loss=%.4f'
                  % (epoch_it, it, loss))

            if epoch_it % epochs_per_save == 0:
                save_dict = {
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch_it': epoch_it,
                    'it': it,
                }
                torch.save(save_dict, os.path.join(out_dir, 'model.pt'))

    def compute_train_loss(self, data, it):
        coords = data["query_point"].float().unsqueeze(0)
        gt_sdf = data["sdf"].float().unsqueeze(0)
        gt_in_contact = data["in_contact"].float().unsqueeze(0)
        wrist_wrench = data["wrist_wrench"].float().unsqueeze(0)
        partial_pointcloud = data["partial_pointcloud"].float().unsqueeze(0)

        # Run model forward.
        z_deform = self.model.encode_pointcloud(partial_pointcloud)
        z_wrench = self.model.encode_wrench(wrist_wrench)
        out_dict = self.model.forward(coords, z_deform, z_wrench)

        # Loss:
        loss_dict = dict()

        # SDF Loss: How accurate are the SDF predictions at each query point.
        sdf_dist = out_dict["sdf_dist"]
        sdf_loss = -sdf_dist.log_prob(gt_sdf).mean()
        loss_dict["sdf_loss"] = sdf_loss

        # Contact prediction loss.
        in_contact_dist = out_dict["in_contact_dist"]
        contact_loss = ncf_losses.heteroscedastic_bce(in_contact_dist, gt_in_contact)
        contact_loss = contact_loss[gt_sdf == 0.0].mean()
        loss_dict["contact_loss"] = contact_loss

        # Network regularization.
        reg_loss = self.model.regularization_loss(out_dict)
        loss_dict["reg_loss"] = reg_loss

        # Calculate total weighted loss.
        loss = 0.0
        for loss_key in loss_dict.keys():
            loss += float(self.train_loss_weights[loss_key]) * loss_dict[loss_key]
        loss_dict["loss"] = loss

        return loss_dict, out_dict
