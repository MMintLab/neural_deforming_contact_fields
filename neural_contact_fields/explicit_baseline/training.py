import os
from tqdm import tqdm

import mmint_utils
import numpy as np
import torch
import open3d as o3d
from tensorboardX import SummaryWriter
from torch import optim

from torch.utils.data import Dataset
from neural_contact_fields.data.tool_dataset import ToolDataset
from neural_contact_fields.neural_contact_field.models.neural_contact_field import NeuralContactField
from neural_contact_fields.training import BaseTrainer
from neural_contact_fields.explicit_baseline.grnet.extensions.chamfer_dist import ChamferDistance


class Trainer(BaseTrainer):

    def __init__(self, cfg, model: NeuralContactField, device):
        super().__init__(cfg, model, device)

        self.model: NeuralContactField = model

    ##########################################################################
    #  Pretraining loop                                                      #
    ##########################################################################

    def pretrain(self, pretrain_dataset: Dataset):
        pass

    def compute_pretrain_loss(self, data, it) -> (dict, dict):

        return {}, {}

    ##########################################################################
    #  Main training loop                                                    #
    ##########################################################################

    def get_loss(self,data_dict):
        # Pull out relevant data.
        query_points = torch.from_numpy(data_dict["query_point"]).to(self.device).float().unsqueeze(0)
        surface_coords_ = torch.from_numpy(data_dict["surface_points"]).to(self.device).float().unsqueeze(0)
        wrist_wrench_ = torch.from_numpy(data_dict["wrist_wrench"]).to(self.device).float().unsqueeze(0)
        gt_sdf = torch.from_numpy(data_dict["sdf"]).to(self.device).float().unsqueeze(0)
        gt_in_contact = torch.from_numpy(data_dict["in_contact"]).to(self.device).float().unsqueeze(0)

#         print(data_dict.keys(), query_points.shape, surface_coords_.shape, gt_sdf.shape)
        cnt_indicator = gt_in_contact[gt_sdf == 0.0]
        cnt_idx = torch.where(gt_in_contact == 1)[1]
        contact_pcd = query_points[:, cnt_idx, :]
#        print(cnt_idx)

        # We assume we know the object code.
        z_wrench_ = self.model.encode_wrench(wrist_wrench_)
        # Predict with updated latents.
        pred_dict_ = self.model.forward(surface_coords_, z_wrench_)


        ## pointcloud reconstruction loss.
        chamfer_dist = ChamferDistance()
        sparse_loss_df = chamfer_dist(surface_coords_ , pred_dict_['sparse_df_cloud'])
        dense_loss_df = chamfer_dist( surface_coords_, pred_dict_['dense_df_ptcloud'])


        sparse_loss_ct = chamfer_dist(pred_dict_['sparse_ct_ptcloud'], contact_pcd)
        dense_loss_ct = chamfer_dist(pred_dict_['dense_ct_ptcloud'], contact_pcd)
        
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(surface_coords_.squeeze().detach().cpu().numpy())
        o3d.io.write_point_cloud('gt_surf.ply',pcl )
        
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(pred_dict_['dense_df_ptcloud'].squeeze().detach().cpu().numpy())
        o3d.io.write_point_cloud('ds_surf.ply', pcl)
        
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(contact_pcd.squeeze().detach().cpu().numpy())
        o3d.io.write_point_cloud('gt_cnt.ply', pcl)
        
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(pred_dict_['dense_ct_ptcloud'].squeeze().detach().cpu().numpy())
        o3d.io.write_point_cloud('ds_cnt.ply', pcl)

        
#         print(pred_dict_['sparse_df_cloud'].shape, pred_dict_['dense_df_ptcloud'].shape, pred_dict_['sparse_ct_ptcloud'].shape, pred_dict_['dense_ct_ptcloud'].shape)
#         print( sparse_loss_df ,dense_loss_df, sparse_loss_ct,+ dense_loss_ct)
        loss_df =  dense_loss_df + dense_loss_ct
        return loss_df




    def validation(self, validation_dataset: ToolDataset, logger: SummaryWriter, epoch_it: int, it: int):
        return
        trial_idcs = np.arange(len(validation_dataset))
        trial_idcs = torch.from_numpy(trial_idcs).to(self.device)

        trial_losses = torch.zeros(len(validation_dataset)).to(self.device)
        for trial_idx in tqdm(trial_idcs):
            data = validation_dataset[trial_idx]

            loss = self.get_loss(data)
            trial_losses[trial_idx] = loss

        # Log average validation loss.
        val_loss = torch.mean(trial_losses)
        logger.add_scalar("val_loss", val_loss, it)


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

        # Get optimizer.
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
                loss = self.get_loss(batch)* 1e4

                self.model.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            print('[Epoch %02d] it=%03d, loss=%.4f'
                  % (epoch_it, it, loss))

            if epoch_it % epochs_per_validation == 0:
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

