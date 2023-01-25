import torch
import torch.nn as nn
from neural_contact_fields.generation import BaseGenerator
from neural_contact_fields.neural_contact_field.models.neural_contact_field import NeuralContactField
from neural_contact_fields.utils.infer_utils import inference_by_optimization
from neural_contact_fields.utils.marching_cubes import create_mesh


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

    def generate_mesh(self, data, meta_data):
        raise Exception("Selected generator does not generate mesh.")


    def generate_pointcloud(self, data_dict, meta_data):
        wrist_wrench_ = torch.from_numpy(data_dict["wrist_wrench"]).to(self.device).float().unsqueeze(0)
        partial_pcd = torch.from_numpy(data_dict["partial_pointcloud"]).to(self.device).float().unsqueeze(0)

#         cnt_indicator = gt_in_contact[gt_sdf == 0.0]
#         cnt_idx = torch.where(gt_in_contact == 1)[1]
#         contact_pcd = query_points[:, cnt_idx, :]

        # We assume we know the object code.
    
        z_wrench_ = self.model.encode_wrench(wrist_wrench_)
        
        # Predict with updated latents.
        pred_dict_ = self.model.forward(partial_pcd, z_wrench_)

        return pred_dict_['sparse_df_cloud'].squeeze().detach().cpu().numpy(), -1

    def generate_contact_patch(self, data_dict, meta_data):
        wrist_wrench_ = torch.from_numpy(data_dict["wrist_wrench"]).to(self.device).float().unsqueeze(0)
        partial_pcd = torch.from_numpy(data_dict["partial_pointcloud"]).to(self.device).float().unsqueeze(0)
#         cnt_indicator = gt_in_contact[gt_sdf == 0.0]
#         cnt_idx = torch.where(gt_in_contact == 1)[1]
#         contact_pcd = query_points[:, cnt_idx, :]

        # We assume we know the object code.
    
        z_wrench_ = self.model.encode_wrench(wrist_wrench_)
        # Predict with updated latents.
        pred_dict_ = self.model.forward(partial_pcd, z_wrench_)

        return pred_dict_['dense_ct_ptcloud'].squeeze().detach().cpu().numpy(), -1
    

    def generate_contact_labels(self, data, meta_data):
        raise Exception("Selected generator does not generate contact label.")

     