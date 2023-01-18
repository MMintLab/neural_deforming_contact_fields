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


class Generator(BaseGenerator):

    def __init__(self, cfg: dict, model: NeuralContactField, device: torch.device):
        self.model: NeuralContactField
        super().__init__(cfg, model, device)

        self.generates_mesh = True
        self.generates_pointcloud = False
        self.generates_contact_patch = False  # TODO: Add this once have a better sense for it.
        self.generates_contact_labels = True

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

    def generate_mesh(self, data, meta_data):
        # Check if we have been provided with the latent already.
        if "latent" in meta_data:
            latent = meta_data["latent"]
        else:
            # Generate deformation code latent.
            z_deform_size = self.model.z_deform_size
            z_deform_, _ = inference_by_optimization(self.model, self.surface_loss_fn, z_deform_size, 1, data,
                                                     device=self.device, verbose=True)
            latent = z_deform_.weight

        # Generate mesh.
        object_idx_ = torch.from_numpy(data["object_idx"]).to(self.device)
        wrist_wrench_ = torch.from_numpy(data["wrist_wrench"]).to(self.device).float().unsqueeze(0)

        # We assume we know the object code.
        z_object = self.model.encode_object(object_idx_)
        z_wrench = self.model.encode_wrench(wrist_wrench_)

        latent_sdf_decoder = LatentSDFDecoder(self.model, z_object, latent, z_wrench)
        mesh = create_mesh(latent_sdf_decoder)

        return mesh, {"latent": latent, "mesh": mesh}

    def generate_pointcloud(self, data, meta_data):
        raise Exception("Selected generator does not generate point clouds.")

    def generate_contact_patch(self, data, meta_data):
        raise Exception("Selected generator does not generate contact patch.")

    def generate_contact_labels(self, data, meta_data):
        # Check if we have been provided with the latent already.
        if "latent" in meta_data:
            latent = meta_data["latent"]
        else:
            # Generate deformation code latent.
            z_deform_size = self.model.z_deform_size
            z_deform_, _ = inference_by_optimization(self.model, self.surface_loss_fn, z_deform_size, 1, data,
                                                     device=self.device, verbose=True)
            latent = z_deform_.weight

        object_idx = torch.from_numpy(data["object_idx"]).to(self.device)
        wrist_wrench = torch.from_numpy(data["wrist_wrench"]).to(self.device).float().unsqueeze(0)
        z_object = self.model.encode_object(object_idx)
        z_wrench = self.model.encode_wrench(wrist_wrench)

        # Get the surface points from the ground truth.
        surface_coords = torch.from_numpy(data["surface_points"]).to(self.device).float().unsqueeze(0)
        surface_pred_dict = self.model.forward(surface_coords, latent, z_object, z_wrench)
        surface_in_contact = surface_pred_dict["in_contact"].squeeze(0)

        return surface_in_contact, {"latent": latent}
