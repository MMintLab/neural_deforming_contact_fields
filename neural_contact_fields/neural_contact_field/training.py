import mmint_utils
import numpy as np
import torch
import torch.nn as nn
from neural_contact_fields.data.tool_dataset import ToolDataset
from neural_contact_fields.neural_contact_field.models.neural_contact_field import NeuralContactField
from neural_contact_fields.training import BaseTrainer
import torch.nn.functional as F
import os
from neural_contact_fields.utils import model_utils
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import Dataset
import neural_contact_fields.loss as ncf_losses


class Trainer(BaseTrainer):

    def __init__(self, cfg, model: NeuralContactField, device):
        super().__init__(cfg, model, device)

        self.model: NeuralContactField = model

    def load_pretrained_model(self, object_code: nn.Embedding, pretrain_file: str, freeze_object_module_weights=False):
        """
        Load pretrained model weights. Optionally freeze object module weights.

        Note: assumes object_module defined in model.
        """
        print("Loading pretrained model weights from local file: %s" % pretrain_file)

        model_utils.load_model({"model": self.model, "object_code": object_code}, pretrain_file)

        # Optionally, we can freeze the pretrained weights. TODO: Make this better.
        if freeze_object_module_weights:
            for param in self.model.object_model.parameters():
                param.requires_grad = False
            object_code.requires_grad_(False)

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
        epoch_it = 0
        it = 0

        # Output + vis directory
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        logger = SummaryWriter(os.path.join(out_dir, 'logs'))

        # Dump config to output directory.
        mmint_utils.dump_cfg(os.path.join(out_dir, 'config.yaml'), self.cfg)

        # Setup the object embedding.
        num_objects = len(pretrain_dataset)
        object_code = nn.Embedding(num_objects, self.cfg["model"]["z_object_size"]).requires_grad_(True).to(self.device)
        nn.init.normal_(object_code.weight, mean=0.0, std=0.1)

        # Get optimizer (TODO: Parameterize?)
        optimizer = optim.Adam([
            {"params": self.model.parameters(), "lr": lr},
            {"params": object_code.parameters(), "lr": lr},
        ])

        # Load model + optimizer if a partially trained copy of it exists.
        self.load_partial_train_model({"model": self.model, "optimizer": optimizer, "object_code": object_code},
                                      out_dir, "pretrain_model.pt")

        # Training loop
        while True:
            epoch_it += 1

            if epoch_it > max_epochs:
                print("Backing up and stopping training. Reached max epochs.")
                save_dict = {
                    'model': self.model.state_dict(),
                    'object_code': object_code.state_dict(),
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

                # Encode the object idx.
                batch["z_object"] = object_code(object_idx).float()

                loss = self.train_step(batch, it, optimizer, logger, self.compute_pretrain_loss)

            print('[Epoch %02d] it=%03d, loss=%.4f'
                  % (epoch_it, it, loss))

            if epoch_it % epochs_per_save == 0:
                save_dict = {
                    'model': self.model.state_dict(),
                    'object_code': object_code.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch_it': epoch_it,
                    'it': it,
                }
                torch.save(save_dict, os.path.join(out_dir, 'pretrain_model.pt'))

        # Backup.
        save_dict = {
            'model': self.model.state_dict(),
            'object_code': object_code.state_dict(),
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
        coords = torch.from_numpy(data["query_point"]).to(self.device).float().unsqueeze(0)
        gt_sdf = torch.from_numpy(data["sdf"]).to(self.device).float().unsqueeze(0)
        gt_normals = torch.from_numpy(data["normals"]).to(self.device).float().unsqueeze(0)
        obj_code = data["z_object"].unsqueeze(0)

        # Forward prediction.
        out_dict = self.model.forward_object_module(coords, obj_code)

        # Calculate Losses:
        loss_dict = dict()

        # SDF Loss: How accurate are the SDF predictions at each query point.
        sdf_loss = ncf_losses.sdf_loss(out_dict["sdf"], gt_sdf)
        loss_dict["sdf_loss"] = sdf_loss

        # Normals loss: are the normals accurate.
        normals_loss = ncf_losses.surface_normal_loss(gt_sdf, gt_normals, out_dict["sdf"], out_dict["query_points"])
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

    def train(self, train_dataset: ToolDataset):
        # Shorthands:
        out_dir = self.cfg['training']['out_dir']
        lr = self.cfg['training']['learning_rate']
        max_epochs = self.cfg['training']['epochs']
        epochs_per_save = self.cfg['training']['epochs_per_save']
        self.train_loss_weights = self.cfg['training']['loss_weights']  # TODO: Better way to set this?
        epoch_it = 0
        it = 0

        # Output + vis directory
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        logger = SummaryWriter(os.path.join(out_dir, 'logs'))

        # Dump config to output directory.
        mmint_utils.dump_cfg(os.path.join(out_dir, 'config.yaml'), self.cfg)

        # Setup the object/trial embeddings.
        num_objects = train_dataset.get_num_objects()
        num_trials = train_dataset.get_num_trials()
        object_code = nn.Embedding(num_objects, self.cfg["model"]["z_object_size"]).requires_grad_(True).to(self.device)
        nn.init.normal_(object_code.weight, mean=0.0, std=0.1)
        trial_code = nn.Embedding(num_trials, self.cfg["model"]["z_deform_size"]).requires_grad_(True).to(self.device)
        nn.init.normal_(trial_code.weight, mean=0.0, std=0.1)

        # Get optimizer (TODO: Parameterize?)
        optim_params = [
            {"params": self.model.parameters(), "lr": lr},
            {"params": trial_code.parameters(), "lr": lr},
        ]
        if self.cfg["training"]["update_object_code"]:
            optim_params.append({"params": object_code.parameters(), "lr": lr})
        optimizer = optim.Adam(optim_params)

        # Load pretrained model, if appropriate.
        pretrain_file = os.path.join(self.cfg["pretraining"]["out_dir"], "pretrain_model.pt")
        self.load_pretrained_model(object_code, pretrain_file, self.cfg['training']['freeze_pretrain_weights'])

        # Load model + optimizer if a partially trained copy of it exists.
        epoch_it, it = self.load_partial_train_model(
            {"model": self.model, "optimizer": optimizer, "object_code": object_code,
             "trial_code": trial_code}, out_dir, "model.pt")

        # Training loop
        while True:
            epoch_it += 1

            if epoch_it > max_epochs:
                print("Backing up and stopping training. Reached max epochs.")
                save_dict = {
                    'model': self.model.state_dict(),
                    'object_code': object_code.state_dict(),
                    'trial_code': trial_code.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch_it': epoch_it,
                    'it': it,
                }
                torch.save(save_dict, os.path.join(out_dir, 'model.pt'))
                break

            loss = None

            for trial_idx in range(len(train_dataset)):
                it += 1

                # For this training, we use just a single example per run.
                batch = train_dataset[trial_idx]
                object_idx_tensor = torch.tensor(batch["object_idx"], device=self.device)
                trial_idx_tensor = torch.tensor(batch["trial_idx"], device=self.device)

                # Encode object idx/trial idx.
                batch["z_object"] = object_code(object_idx_tensor).float()
                batch["z_trial"] = trial_code(trial_idx_tensor).float()

                loss = self.train_step(batch, it, optimizer, logger, self.compute_train_loss)

            print('[Epoch %02d] it=%03d, loss=%.4f'
                  % (epoch_it, it, loss))

            if epoch_it % epochs_per_save == 0:
                save_dict = {
                    'model': self.model.state_dict(),
                    'object_code': object_code.state_dict(),
                    'trial_code': trial_code.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch_it': epoch_it,
                    'it': it,
                }
                torch.save(save_dict, os.path.join(out_dir, 'model.pt'))

        # Backup.
        save_dict = {
            'model': self.model.state_dict(),
            'object_code': object_code.state_dict(),
            'trial_code': trial_code.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch_it': epoch_it,
            'it': it,
        }
        torch.save(save_dict, os.path.join(out_dir, 'model.pt'))

    def compute_train_loss(self, data, it) -> (dict, dict):
        """
        Get loss over provided batch of data.

        Args:
        - data (dict): data dictionary
        """
        obj_code = data["z_object"].unsqueeze(0)
        trial_code = data["z_trial"].unsqueeze(0)
        coords = torch.from_numpy(data["query_point"]).to(self.device).float().unsqueeze(0)
        gt_sdf = torch.from_numpy(data["sdf"]).to(self.device).float().unsqueeze(0)
        gt_normals = torch.from_numpy(data["normals"]).to(self.device).float().unsqueeze(0)
        gt_in_contact = torch.from_numpy(data["in_contact"]).to(self.device).float().unsqueeze(0)

        out_dict = self.model.forward(coords, trial_code, obj_code)

        # Loss:
        loss_dict = dict()

        # SDF Loss: How accurate are the SDF predictions at each query point.
        sdf_loss = ncf_losses.sdf_loss(out_dict["sdf"], gt_sdf)
        loss_dict["sdf_loss"] = sdf_loss

        # Normals loss: are the normals accurate.
        normals_loss = ncf_losses.surface_normal_loss(gt_sdf, gt_normals, out_dict["sdf"], out_dict["query_points"])
        loss_dict["normals_loss"] = normals_loss

        # Latent embedding loss: well-formed embedding.
        embedding_loss = ncf_losses.l2_loss(out_dict["embedding"], squared=True)
        loss_dict["embedding_loss"] = embedding_loss

        # Loss on deformation field.
        def_loss = ncf_losses.l2_loss(out_dict["pred_deform"], squared=True)
        loss_dict["def_loss"] = def_loss

        # Network regularization.
        reg_loss = self.model.regularization_loss(out_dict)
        loss_dict["reg_loss"] = reg_loss

        # Contact prediction loss.
        contact_loss = F.binary_cross_entropy_with_logits(out_dict["in_contact_logits"], gt_in_contact)
        loss_dict["contact_loss"] = contact_loss

        # Calculate total weighted loss.
        loss = 0.0
        for loss_key in loss_dict.keys():
            loss += float(self.train_loss_weights[loss_key]) * loss_dict[loss_key]
        loss_dict["loss"] = loss

        return loss_dict, out_dict
