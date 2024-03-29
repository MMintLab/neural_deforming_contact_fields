import mmint_utils
import numpy as np
import torch
import torch.nn as nn
from neural_contact_fields.data.tool_dataset import ToolDataset
from neural_contact_fields.encoder_ndcf.models.neural_contact_field import NeuralContactField
from neural_contact_fields.training import BaseTrainer
import torch.nn.functional as F
import os
from neural_contact_fields.utils import model_utils
from neural_contact_fields.utils.infer_utils import inference_by_optimization
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import Dataset
import neural_contact_fields.loss as ncf_losses
from tqdm import tqdm


class Trainer(BaseTrainer):

    def __init__(self, cfg, model: NeuralContactField, device):
        super().__init__(cfg, model, device)

        self.model: NeuralContactField = model

    ##########################################################################
    #  Pretraining loop                                                      #
    ##########################################################################

    def pretrain(self, pretrain_dataset: Dataset):
        # Shorthands:
        out_dir = self.cfg['pretraining']['out_dir']
        lr = self.cfg['pretraining']['learning_rate']
        max_epochs = self.cfg['pretraining']['epochs']
        epochs_per_save = self.cfg['pretraining']['epochs_per_save']
        self.pretrain_loss_weights = self.cfg['pretraining']['loss_weights']  # TODO: Better way to set this?

        # Output + vis directory
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        logger = SummaryWriter(os.path.join(out_dir, 'logs/pretrain'))

        # Dump config to output directory.
        mmint_utils.dump_cfg(os.path.join(out_dir, 'config.yaml'), self.cfg)

        # Get optimizer (TODO: Parameterize?)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Load model + optimizer if a partially trained copy of it exists.
        epoch_it, it = self.load_partial_train_model(
            {"model": self.model, "optimizer": optimizer},
            out_dir, "pretrain_model.pt")

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
                torch.save(save_dict, os.path.join(out_dir, 'pretrain_model.pt'))
                break

            loss = None

            object_idcs = np.arange(len(pretrain_dataset))
            np.random.shuffle(object_idcs)
            object_idcs = torch.from_numpy(object_idcs).to(self.device)

            for object_idx in object_idcs:
                it += 1

                # For this training, we use just a single example per run.
                batch = pretrain_dataset[object_idx]
                loss = self.train_step(batch, it, optimizer, logger, self.compute_pretrain_loss)

            print('[Epoch %02d] it=%03d, loss=%.4f'
                  % (epoch_it, it, loss))

            if epoch_it % epochs_per_save == 0:
                save_dict = {
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch_it': epoch_it,
                    'it': it,
                }
                torch.save(save_dict, os.path.join(out_dir, 'pretrain_model.pt'))

    def compute_pretrain_loss(self, data, it) -> (dict, dict):
        """
        Get loss over provided batch of data.

        Args:
        - data (dict): data dictionary
        """
        # Load data from batch.
        object_idx = torch.from_numpy(data["object_idx"]).to(self.device)
        coords = torch.from_numpy(data["query_point"]).to(self.device).float().unsqueeze(0)
        gt_sdf = torch.from_numpy(data["sdf"]).to(self.device).float().unsqueeze(0)
        gt_normals = torch.from_numpy(data["normals"]).to(self.device).float().unsqueeze(0)

        # Forward prediction.
        z_object = self.model.encode_object(object_idx)
        out_dict = self.model.forward_object_module(coords, z_object)

        # Calculate Losses:
        loss_dict = dict()

        # SDF Loss: How accurate are the SDF predictions at each query point.
        sdf_loss = ncf_losses.sdf_loss(out_dict["sdf"], gt_sdf)
        loss_dict["sdf_loss"] = sdf_loss

        # Normals loss: are the normals accurate.
        normals_loss = ncf_losses.surface_normal_loss(gt_sdf, gt_normals, out_dict["normals"])
        loss_dict["normals_loss"] = normals_loss

        # Latent embedding loss: well-formed embedding.
        embedding_loss = ncf_losses.l2_loss(out_dict["embedding"], squared=True)
        loss_dict["embedding_loss"] = embedding_loss

        # Network regularization.
        reg_loss = self.model.object_module_regularization_loss(out_dict)
        loss_dict["reg_loss"] = reg_loss

        # Calculate total weighted loss.
        loss = 0.0
        for loss_key in loss_dict.keys():
            loss += float(self.pretrain_loss_weights[loss_key]) * loss_dict[loss_key]
        loss_dict["loss"] = loss

        return loss_dict, out_dict

    ##########################################################################
    #  Main training loop                                                    #
    ##########################################################################

    def surface_loss_fn(self, model, latent, data_dict):
        # Pull out relevant data.
        object_idx_ = torch.from_numpy(data_dict["object_idx"]).to(self.device)
        surface_coords_ = torch.from_numpy(data_dict["surface_points"]).to(self.device).float().unsqueeze(0)
        wrist_wrench_ = torch.from_numpy(data_dict["wrist_wrench"]).to(self.device).float().unsqueeze(0)

        # We assume we know the object code.
        z_object_ = self.model.encode_object(object_idx_)
        z_wrench_ = self.model.encode_wrench(wrist_wrench_)

        # Predict with updated latents.
        pred_dict_ = self.model.forward(surface_coords_, latent, z_object_, z_wrench_)

        # Loss: all points on surface should have SDF = 0.0.
        loss = torch.mean(torch.abs(pred_dict_["sdf"]))

        return loss

    def validation(self, validation_dataset: ToolDataset, logger: SummaryWriter, epoch_it: int, it: int):
        trial_idcs = np.arange(len(validation_dataset))
        trial_idcs = torch.from_numpy(trial_idcs).to(self.device)

        def validation_loss_fn(model, latent, data_dict, device):
            loss_dict, _ = self.compute_train_loss_from_latent(data_dict, latent)
            return loss_dict["loss"]

        trial_losses = torch.zeros(len(validation_dataset)).to(self.device)
        trial_sf_losses = torch.zeros(len(validation_dataset)).to(self.device)
        for trial_idx in tqdm(trial_idcs):
            data = validation_dataset[trial_idx]

            # Run inference to recover latent for validation example.
            _, final_loss = inference_by_optimization(self.model, validation_loss_fn, self.model.z_deform_size, 1, data,
                                                      device=self.device, verbose=False)
            trial_losses[trial_idx] = final_loss

            # Run inference but only with surface points.
            # _, final_sf_loss = inference_by_optimization(self.model, self.surface_loss_fn, self.model.z_deform_size,
            #                                              1, data, device=self.device, verbose=False)
            # trial_sf_losses[trial_idx] = final_sf_loss

        # Log average validation loss.
        val_loss = torch.mean(trial_losses)
        logger.add_scalar("val_loss", val_loss, it)
        val_sf_loss = torch.mean(trial_sf_losses)
        logger.add_scalar("val_sf_loss", val_sf_loss, it)

    def train(self, train_dataset: ToolDataset, validation_dataset: ToolDataset):
        # Shorthands:
        out_dir = self.cfg['training']['out_dir']
        lr = self.cfg['training']['learning_rate']
        max_epochs = self.cfg['training']['epochs']
        epochs_per_save = self.cfg['training']['epochs_per_save']
        epochs_per_validation = self.cfg['training']['epochs_per_validation']
        self.train_loss_weights = self.cfg['training']['loss_weights']  # TODO: Better way to set this?

        # Output + vis directory
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        logger = SummaryWriter(os.path.join(out_dir, 'logs/train'))

        # Dump config to output directory.
        mmint_utils.dump_cfg(os.path.join(out_dir, 'config.yaml'), self.cfg)

        # Get optimizer (TODO: Parameterize?)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Load pretrained model, if appropriate.
        pretrain_file = os.path.join(self.cfg["pretraining"]["out_dir"], "pretrain_model.pt")
        self.model.load_pretrained_model(pretrain_file, self.cfg['training']['load_pretrain'])

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

            if epoch_it % epochs_per_validation == 0 and validation_dataset is not None:
                print("Validating model...")
                self.validation(validation_dataset, logger, epoch_it, it)

            if epoch_it % epochs_per_save == 0:
                save_dict = {
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch_it': epoch_it,
                    'it': it,
                }
                torch.save(save_dict, os.path.join(out_dir, 'model.pt'))

    def compute_train_loss_from_latent(self, data, z_trial):
        object_idx = data["object_idx"]
        coords = data["query_point"].float().unsqueeze(0)
        gt_sdf = data["sdf"].float().unsqueeze(0)
        gt_normals = data["normals"].float().unsqueeze(0)
        gt_in_contact = data["in_contact"].float().unsqueeze(0)
        nominal_coords = data["nominal_query_point"].float().unsqueeze(0)
        nominal_sdf = data["nominal_sdf"].float().unsqueeze(0)
        wrist_wrench = data["wrist_wrench"].float().unsqueeze(0)

        # Run model forward.
        z_object = self.model.encode_object(object_idx)
        z_wrench = self.model.encode_wrench(wrist_wrench)
        out_dict = self.model.forward(coords, z_trial, z_object, z_wrench)

        # Loss:
        loss_dict = dict()

        # SDF Loss: How accurate are the SDF predictions at each query point.
        sdf_loss = ncf_losses.sdf_loss(out_dict["sdf"], gt_sdf)
        loss_dict["sdf_loss"] = sdf_loss

        # Normals loss: are the normals accurate.
        normals_loss = ncf_losses.surface_normal_loss(gt_sdf, gt_normals, out_dict["normals"])
        loss_dict["normals_loss"] = normals_loss

        # Latent embedding loss: well-formed embedding.
        embedding_loss = ncf_losses.l2_loss(out_dict["embedding"], squared=True)
        loss_dict["embedding_loss"] = embedding_loss

        # Loss on deformation field.
        def_loss = ncf_losses.l2_loss(out_dict["deform"], squared=True)
        loss_dict["def_loss"] = def_loss

        # Network regularization.
        reg_loss = self.model.regularization_loss(out_dict)
        loss_dict["reg_loss"] = reg_loss

        # Contact prediction loss.
        contact_loss = F.binary_cross_entropy_with_logits(out_dict["in_contact_logits"][gt_sdf == 0.0],
                                                          gt_in_contact[gt_sdf == 0.0])
        loss_dict["contact_loss"] = contact_loss

        # Chamfer distance loss.
        chamfer_loss = ncf_losses.surface_chamfer_loss(nominal_coords, nominal_sdf, gt_sdf, out_dict["nominal"])
        loss_dict["chamfer_loss"] = chamfer_loss

        # Calculate total weighted loss.
        loss = 0.0
        for loss_key in loss_dict.keys():
            loss += float(self.train_loss_weights[loss_key]) * loss_dict[loss_key]
        loss_dict["loss"] = loss

        return loss_dict, out_dict

    def compute_train_loss(self, data, it) -> (dict, dict):
        """
        Get loss over provided batch of data.

        Args:
        - data (dict): data dictionary
        """
        object_idx = data["object_idx"]
        trial_idx = data["trial_idx"]
        partial_pointcloud = data["partial_pointcloud"].float().unsqueeze(0)

        # Run model forward.
        # _, z_trial = self.model.encode_trial(object_idx, trial_idx)
        z_trial = self.model.encode_pointcloud(partial_pointcloud)

        return self.compute_train_loss_from_latent(data, z_trial)
