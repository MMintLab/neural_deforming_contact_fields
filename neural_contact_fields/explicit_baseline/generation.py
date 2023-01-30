import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
from neural_contact_fields.generation import BaseGenerator
from neural_contact_fields.neural_contact_field.models.neural_contact_field import NeuralContactField
from neural_contact_fields.utils.infer_utils import inference_by_optimization
from neural_contact_fields.utils.marching_cubes import create_mesh
from neural_contact_fields.explicit_baseline.grnet.extensions.chamfer_dist import ChamferDistance

class LatentSDFDecoder(nn.Module):
    """
    Decoder when given latent variables.
    """

    def __init__(self, model: NeuralContactField, z_object, z_deform, z_wrench):
        super().__init__()
        self.model: NeuralContactField = model
        self.model.eval()
        self.z_object = z_object
        self.z_deform = z_deform
        self.z_wrench = z_wrench

    def forward(self, query_points: torch.Tensor):
        out_dict = self.model.forward(query_points.unsqueeze(0), self.z_deform, self.z_object, self.z_wrench)
        return out_dict["sdf"].squeeze(0)


def surface_loss_fn(model, latent, data_dict, device):
    # Pull out relevant data.
    object_idx_ = torch.from_numpy(data_dict["object_idx"]).to(device)
    surface_coords_ = torch.from_numpy(data_dict["surface_points"]).to(device).float().unsqueeze(0)
    wrist_wrench_ = torch.from_numpy(data_dict["wrist_wrench"]).to(device).float().unsqueeze(0)

    # We assume we know the object code.
    z_object_ = model.encode_object(object_idx_)
    z_wrench_ = model.encode_wrench(wrist_wrench_)

    # Predict with updated latents.
    pred_dict_ = model.forward(surface_coords_, latent, z_object_, z_wrench_)

    # Loss: all points on surface should have SDF = 0.0.
    loss = torch.mean(torch.abs(pred_dict_["sdf"]))

    return loss


class Generator(BaseGenerator):

    def __init__(self, cfg: dict, model: NeuralContactField, device: torch.device):
        self.model: NeuralContactField
        super().__init__(cfg, model, device)

        self.generates_mesh = False
        self.generates_pointcloud = True
        self.generates_contact_patch = True  # TODO: Add this once have a better sense for it.
        self.generates_contact_labels = False
        self.output_length = 10000

    def generate_mesh(self, data, meta_data):
        raise Exception("Selected generator does not generate mesh.")

    def generate_pointcloud(self, data_dict, meta_data):
        wrist_wrench_ = torch.from_numpy(data_dict["wrist_wrench"]).to(self.device).float().unsqueeze(0)
        partial_pcd = torch.from_numpy(data_dict["partial_pointcloud"]).to(self.device).float().unsqueeze(0)
        surface_coords_ = torch.from_numpy(data_dict["surface_points"]).to(self.device).float().unsqueeze(0)
        
        z_wrench_ = self.model.encode_wrench(wrist_wrench_)
        
        # Predict with updated latents.
        pred_dict_ = self.model.forward(partial_pcd, z_wrench_)


        chamfer_dist = ChamferDistance()
        gt_idx = np.random.choice ( np.arange(len(surface_coords_[0])), self.output_length , replace = False)
        est_idx = np.random.choice ( np.arange(len(pred_dict_['dense_df_ptcloud'][0])), self.output_length , replace = False)

        dense_loss_df = chamfer_dist(pred_dict_['dense_df_ptcloud'][:,est_idx,:], surface_coords_[:,gt_idx,:])

        return pred_dict_['dense_df_ptcloud'].squeeze().detach().cpu().numpy(), dense_loss_df.detach().cpu()
    #
    def generate_contact_patch(self, data_dict, meta_data):
        wrist_wrench_ = torch.from_numpy(data_dict["wrist_wrench"]).to(self.device).float().unsqueeze(0)
        partial_pcd = torch.from_numpy(data_dict["partial_pointcloud"]).to(self.device).float().unsqueeze(0)
        query_points = torch.from_numpy(data_dict["query_point"]).to(self.device).float().unsqueeze(0)
        gt_in_contact = torch.from_numpy(data_dict["in_contact"]).to(self.device).float().unsqueeze(0)
        cnt_idx = torch.where(gt_in_contact == 1)[1]
        contact_pcd = query_points[:, cnt_idx, :]


        # We assume we know the object code.
        z_wrench_ = self.model.encode_wrench(wrist_wrench_)
        
        
        # Predict with updated latents.
        pred_dict_ = self.model.forward(partial_pcd, z_wrench_)
        
                
        chamfer_dist = ChamferDistance()
        gt_idx = np.random.choice ( np.arange(len(contact_pcd[0])),  500, replace = False)
        est_idx = np.random.choice ( np.arange(len(pred_dict_['dense_ct_ptcloud'][0])),  500, replace = False)

        dense_loss_df = chamfer_dist( pred_dict_['dense_ct_ptcloud'][:,est_idx,:], contact_pcd[:,gt_idx,:])

        return pred_dict_['dense_ct_ptcloud'].squeeze().detach().cpu().numpy(), dense_loss_df.detach().cpu()
    

    def generate_contact_labels(self, data, meta_data):
        raise Exception("Selected generator does not generate contact label.")

     
