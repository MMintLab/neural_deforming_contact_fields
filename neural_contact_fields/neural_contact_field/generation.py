import numpy as np
import torch
import torch.nn as nn
from neural_contact_fields.generation import BaseGenerator
from neural_contact_fields.neural_contact_field.models.neural_contact_field import NeuralContactField
from neural_contact_fields.utils import mesh_utils
from neural_contact_fields.utils.infer_utils import inference_by_optimization
from neural_contact_fields.utils.marching_cubes import create_mesh
import neural_contact_fields.loss as ncf_losses


class LatentSDFDecoder(nn.Module):
    """
    Decoder when given latent variables.
    """

    def __init__(self, model: NeuralContactField, z_deform, z_object, z_wrench):
        super().__init__()
        self.model: NeuralContactField = model
        self.model.eval()
        self.z_deform = z_deform
        self.z_object = z_object
        self.z_wrench = z_wrench

    def forward(self, query_points: torch.Tensor):
        out_dict = self.model.forward(query_points.unsqueeze(0), self.z_deform, self.z_object, self.z_wrench)
        return out_dict["sdf"].squeeze(0)


def get_surface_loss_fn(embed_weight: float, def_weight: float):
    def surface_loss_fn(model, latent, data_dict, device):
        # Pull out relevant data.
        object_idx_ = torch.from_numpy(data_dict["object_idx"]).to(device)
        surface_coords_ = torch.from_numpy(data_dict["partial_pointcloud"]).to(device).float().unsqueeze(0)
        wrist_wrench_ = torch.from_numpy(data_dict["wrist_wrench"]).to(device).float().unsqueeze(0)

        # We assume we know the object code.
        z_object_ = model.encode_object(object_idx_)
        z_wrench_ = model.encode_wrench(wrist_wrench_)

        # Predict with updated latents.
        pred_dict_ = model.forward(surface_coords_, latent, z_object_, z_wrench_)

        # loss = 0.0

        # Loss: all points on surface should have SDF = 0.0.
        sdf_loss = torch.mean(torch.abs(pred_dict_["sdf"]))

        # Latent embedding loss: shouldn't drift too far from data.
        embedding_loss = ncf_losses.l2_loss(pred_dict_["embedding"], squared=True)

        # Prefer smaller deformations.
        def_loss = ncf_losses.l2_loss(pred_dict_["deform"], squared=True)

        loss = sdf_loss + (embed_weight * embedding_loss) + (def_weight * def_loss)
        return loss

    return surface_loss_fn


class Generator(BaseGenerator):

    def __init__(self, cfg: dict, model: NeuralContactField, generation_cfg: dict, device: torch.device):
        self.model: NeuralContactField
        super().__init__(cfg, model, device)

        self.generates_mesh = True
        self.generates_pointcloud = False
        self.generates_contact_patch = True
        self.generates_contact_labels = False

        self.contact_threshold = generation_cfg.get("contact_threshold", 0.5)
        self.embed_weight = generation_cfg.get("embed_weight", 1.0)
        self.def_weight = generation_cfg.get("def_weight", 0.0)
        self.contact_patch_size = generation_cfg.get("contact_patch_size", 10000)

    def generate_mesh(self, data, meta_data):
        # Check if we have been provided with the latent already.
        if "latent" in meta_data:
            latent = meta_data["latent"]
        else:
            # Generate deformation code latent.
            z_deform_size = self.model.z_deform_size
            z_deform_, _ = inference_by_optimization(self.model,
                                                     get_surface_loss_fn(self.embed_weight, self.def_weight),
                                                     z_deform_size, 1, data,
                                                     device=self.device, verbose=True)
            latent = z_deform_.weight

        # Generate mesh.
        object_idx_ = torch.from_numpy(data["object_idx"]).to(self.device)
        wrist_wrench_ = torch.from_numpy(data["wrist_wrench"]).to(self.device).float().unsqueeze(0)

        # We assume we know the object code.
        z_object = self.model.encode_object(object_idx_)
        z_wrench = self.model.encode_wrench(wrist_wrench_)

        latent_sdf_decoder = LatentSDFDecoder(self.model, latent, z_object, z_wrench)
        mesh = create_mesh(latent_sdf_decoder)

        return mesh, {"latent": latent, "mesh": mesh}

    def generate_pointcloud(self, data, meta_data):
        raise Exception("Selected generator does not generate point clouds.")

    def generate_contact_patch(self, data, meta_data):
        # If mesh has not already been generated, go ahead and generate it.
        if "mesh" not in meta_data:
            mesh, mesh_meta_data = self.generate_mesh(data, meta_data)
            latent = mesh_meta_data["latent"]
        else:
            mesh = meta_data["mesh"]
            latent = meta_data["latent"]

        object_idx_ = torch.from_numpy(data["object_idx"]).to(self.device)
        wrist_wrench_ = torch.from_numpy(data["wrist_wrench"]).to(self.device).float().unsqueeze(0)

        # We assume we know the object code.
        z_object = self.model.encode_object(object_idx_)
        z_wrench = self.model.encode_wrench(wrist_wrench_)

        # We sample contact patch by sampling points on the surface of the mesh and finding any on contact until desired
        # number is met.
        contact_patch = []
        num_contacts_found = 0

        iters = 0
        while num_contacts_found < self.contact_patch_size:
            iters += 1
            surface_point_samples_np = mesh.sample(100000)
            surface_point_samples = torch.from_numpy(surface_point_samples_np).float().to(self.device).unsqueeze(0)
            surface_pred_dict = self.model.forward(surface_point_samples, latent, z_object, z_wrench)
            surface_in_contact = surface_pred_dict["in_contact"].squeeze(0)
            surface_binary = surface_in_contact > self.contact_threshold
            surface_binary_np = surface_binary.cpu().numpy()

            contact_patch.append(surface_point_samples_np[surface_binary_np])
            num_contacts_found += surface_binary_np.sum()

            if iters > 100:
                break
        contact_patch = np.concatenate(contact_patch, axis=0)

        # We may have estimated more than desired, so we select down.
        contact_patch = contact_patch[:self.contact_patch_size]

        return contact_patch, {}

    def generate_contact_labels(self, data, meta_data):
        # Check if we have been provided with the latent already.
        if "latent" in meta_data:
            latent = meta_data["latent"]
        else:
            # Generate deformation code latent.
            z_deform_size = self.model.z_deform_size
            z_deform_, _ = inference_by_optimization(self.model,
                                                     get_surface_loss_fn(self.embed_weight, self.def_weight),
                                                     z_deform_size, 1, data,
                                                     device=self.device, verbose=False)
            latent = z_deform_.weight

        object_idx = torch.from_numpy(data["object_idx"]).to(self.device)
        wrist_wrench = torch.from_numpy(data["wrist_wrench"]).to(self.device).float().unsqueeze(0)
        z_object = self.model.encode_object(object_idx)
        z_wrench = self.model.encode_wrench(wrist_wrench)

        # Get the surface points from the ground truth.
        surface_coords = torch.from_numpy(data["surface_points"]).to(self.device).float().unsqueeze(0)
        surface_pred_dict = self.model.forward(surface_coords, latent, z_object, z_wrench)
        surface_in_contact_logits = surface_pred_dict["in_contact_logits"].squeeze(0)
        surface_in_contact = surface_pred_dict["in_contact"].squeeze(0)
        surface_binary = surface_in_contact > self.contact_threshold

        return {"contact_labels": surface_binary, "contact_prob": surface_in_contact,
                "contact_logits": surface_in_contact_logits}, {"latent": latent}
