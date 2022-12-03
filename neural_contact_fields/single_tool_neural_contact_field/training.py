import torch
from neural_contact_fields.training import BaseTrainer
import torch.nn.functional as F
import os
import mmint_utils
from neural_contact_fields.utils import model_utils
from tensorboardX import SummaryWriter
import torch.optim as optim


class Trainer(BaseTrainer):

    def __init__(self, cfg, model, device=None):
        """
        Args:
        - model (nn.Module): model
        - optimizer (optimizer): pytorch optimizer
        - logger (tensorboardX.SummaryWriter): logger for tensorboard
        - vis_dir (str): vis directory
        - device (device): pytorch device
        """
        self.model = model
        self.device = device

        # Shorthands:
        self.out_dir = cfg['training']['out_dir']
        self.lr = cfg['training']['learning_rate']
        self.print_every = cfg['training']['print_every']
        self.max_epochs = cfg['training']['epochs']
        self.loss_weights = cfg['training']['loss_weights']
        self.epoch_it = 0
        self.it = 0

        # Output + vis directory
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        self.logger = SummaryWriter(os.path.join(self.out_dir, 'logs'))

        # Dump config to output directory.
        mmint_utils.dump_cfg(os.path.join(self.out_dir, 'config.yaml'), cfg)

        # Get optimizer (TODO: Parameterize?)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # Load pretrained model, if appropriate.
        self.pretrain_file = os.path.join(self.out_dir, cfg['training']['pretrain_file'])
        self.load_pretrained_model(cfg['training']['freeze_pretrain_weights'])

        # Load model + optimizer if a partially trained copy of it exists.
        self.load_partial_train_model()

    def load_partial_train_model(self):
        model_dict = {
            'model': self.model,
            'optimizer': self.optimizer,
        }
        model_file = os.path.join(self.out_dir, 'model.pt')
        load_dict = model_utils.load_model(model_dict, model_file)
        self.epoch_it = load_dict.get('epoch_it', -1)
        self.it = load_dict.get('it', -1)

    ##########################################################################
    #  Pretraining loop                                                      #
    ##########################################################################

    def load_pretrained_model(self, freeze_pretrain_weights=False):
        # Load model pretrain weights, if exists.
        if self.pretrain_file is not None:
            print("Loading pretrained model weights from local file: %s" % self.pretrain_file)
            pretrain_state_dict = torch.load(self.pretrain_file, map_location='cpu')

            # Here, we only load object module weights.
            object_module_keys = [key for key in pretrain_state_dict["model"].keys() if "object_module" in key]
            object_module_dict = {k: pretrain_state_dict["model"][k] for k in object_module_keys}
            self.model.load_state_dict(object_module_dict, strict=False)

            # Optionally, we can freeze the pretrained weights.
            if freeze_pretrain_weights:
                for param in self.model.object_module.parameters():
                    param.requires_grad = False

    def pretrain(self, pretrain_dataset):
        raise NotImplementedError()

    ##########################################################################
    #  Main training loop                                                    #
    ##########################################################################

    def train(self, train_dataset):
        # Training loop
        while True:
            self.epoch_it += 1

            if self.epoch_it > self.max_epochs:
                print("Backing up and stopping training. Reached max epochs.")
                save_dict = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch_it': self.epoch_it,
                    'it': self.it,
                }
                torch.save(save_dict, os.path.join(self.out_dir, 'model.pt'))
                break

            loss = None
            for trial_idx in range(train_dataset.num_trials):
                self.it += 1

                batch = train_dataset.get_all_points_for_trial_batch(-1, trial_idx)
                loss = self.train_step(batch, self.it)

            print('[Epoch %02d] it=%03d, loss=%.4f'
                  % (self.epoch_it, self.it, loss))

            # Backup.
            save_dict = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch_it': self.epoch_it,
                'it': self.it,
            }
            torch.save(save_dict, os.path.join(self.out_dir, 'model.pt'))

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
        surface_query_points = sdf == 0.0
        in_contact = data["in_contact"].to(device) > 0.5
        in_contact_float = torch.clone(in_contact).float()
        force = data["force"].float().to(device)

        pred_dict = self.model.forward(trial_idx, query_point)

        # Apply losses.

        # Apply L2 loss on trial embeddings.
        embed_loss = torch.linalg.norm(pred_dict["z"]).mean()
        loss_dict["embed_loss"] = embed_loss  # TODO: This is not being used.

        # We apply the SDF loss to every point in space.
        sdf_loss = F.l1_loss(pred_dict["sdf"], sdf, reduction="mean")
        loss_dict["sdf_loss"] = sdf_loss

        # Apply L2 loss on deformation.
        def_loss = torch.linalg.norm(pred_dict["delta_coords"]).mean()
        loss_dict["def_loss"] = def_loss

        # Next, for all points *on the surface* we predict the contact probability.
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
