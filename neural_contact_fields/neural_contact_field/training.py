import mmint_utils
import torch
from neural_contact_fields.training import BaseTrainer
import torch.nn.functional as F
import os
from neural_contact_fields.utils import model_utils
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import Dataset


class Trainer(BaseTrainer):

    def load_pretrained_model(self, pretrain_file: str, freeze_object_module_weights=False):
        """
        Load pretrained model weights. Optionally freeze object module weights.

        Note: assumes object_module defined in model.
        """
        print("Loading pretrained model weights from local file: %s" % pretrain_file)
        pretrain_state_dict = torch.load(pretrain_file, map_location='cpu')

        # Here, we only load object module weights.
        object_module_keys = [key for key in pretrain_state_dict["model"].keys() if "object_module" in key]
        object_module_dict = {k: pretrain_state_dict["model"][k] for k in object_module_keys}
        self.model.load_state_dict(object_module_dict, strict=False)

        # Optionally, we can freeze the pretrained weights.
        if freeze_object_module_weights:
            for param in self.model.object_module.parameters():
                param.requires_grad = False

    ##########################################################################
    #  Pretraining loop                                                      #
    ##########################################################################

    def pretrain(self, pretrain_dataset: Dataset):
        # Shorthands:
        out_dir = self.cfg['pretraining']['out_dir']
        lr = self.cfg['pretraining']['learning_rate']
        max_epochs = self.cfg['pretraining']['epochs']
        self.pretrain_loss_weights = self.cfg['pretraining']['loss_weights']  # TODO: Better way to set this?
        epoch_it = 0
        it = 0

        # Output + vis directory
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        logger = SummaryWriter(os.path.join(out_dir, 'logs'))

        # Dump config to output directory.
        mmint_utils.dump_cfg(os.path.join(out_dir, 'config.yaml'), self.cfg)

        # Get optimizer (TODO: Parameterize?)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Load model + optimizer if a partially trained copy of it exists.
        self.load_partial_train_model(optimizer, out_dir, "pretrain_model.pt")

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

            for trial_idx in range(len(pretrain_dataset)):
                it += 1

                # For this training, we use just a single example per run.
                batch = pretrain_dataset[trial_idx]
                loss = self.train_step(batch, it, optimizer, logger, self.compute_pretrain_loss)

            print('[Epoch %02d] it=%03d, loss=%.4f'
                  % (epoch_it, it, loss))

        # Backup.
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
        loss_dict = dict()
        out_dict = dict()

        loss_dict["loss"] = 0.0

        return loss_dict, out_dict

    ##########################################################################
    #  Main training loop                                                    #
    ##########################################################################

    def train(self, train_dataset: Dataset):
        # Shorthands:
        out_dir = self.cfg['training']['out_dir']
        lr = self.cfg['training']['learning_rate']
        max_epochs = self.cfg['training']['epochs']
        self.train_loss_weights = self.cfg['training']['loss_weights']  # TODO: Better way to set this?
        epoch_it = 0
        it = 0

        # Output + vis directory
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        logger = SummaryWriter(os.path.join(out_dir, 'logs'))

        # Dump config to output directory.
        mmint_utils.dump_cfg(os.path.join(out_dir, 'config.yaml'), self.cfg)

        # Get optimizer (TODO: Parameterize?)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Load pretrained model, if appropriate.
        pretrain_file = os.path.join(out_dir, self.cfg['training']['pretrain_file'])
        self.load_pretrained_model(pretrain_file, self.cfg['training']['freeze_pretrain_weights'])

        # Load model + optimizer if a partially trained copy of it exists.
        self.load_partial_train_model(optimizer, out_dir, "model.pt")

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

            for trial_idx in range(len(train_dataset)):
                it += 1

                # For this training, we use just a single example per run.
                batch = train_dataset[trial_idx]
                loss = self.train_step(batch, it, optimizer, logger, self.compute_train_loss)

            print('[Epoch %02d] it=%03d, loss=%.4f'
                  % (epoch_it, it, loss))

        # Backup.
        save_dict = {
            'model': self.model.state_dict(),
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
        loss_dict = dict()
        out_dict = dict()

        loss_dict["loss"] = 0.0

        return loss_dict, out_dict
