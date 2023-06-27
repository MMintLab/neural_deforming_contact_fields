import time

import numpy as np
import torch
from neural_contact_fields.generation import BaseGenerator
from neural_contact_fields.undcf.models.undcf import UNDCF
from neural_contact_fields.utils.infer_utils import inference_by_optimization
from neural_contact_fields.utils.marching_cubes import create_mesh


class Generator(BaseGenerator):

    def __init__(self, cfg: dict, model: UNDCF, generation_cfg: dict, device: torch.device):
        self.model: UNDCF
        super().__init__(cfg, model, device)
        self.model.eval()

        self.generates_nominal_mesh = False
        self.generates_mesh = True
        self.generates_pointcloud = False
        self.generates_contact_patch = True
        self.generates_contact_labels = False
        self.generates_iou_labels = False

        self.contact_threshold = generation_cfg.get("contact_threshold", 0.5)
        self.embed_weight = generation_cfg.get("embed_weight", 0.0)
        self.contact_patch_size = generation_cfg.get("contact_patch_size", 10000)

        self.mesh_resolution = generation_cfg.get("mesh_resolution", 64)

        self.num_latent = int(generation_cfg.get("num_latent", 1))
        self.iter_limit = int(generation_cfg.get("iter_limit", 150))
        self.conv_eps = float(generation_cfg.get("conv_eps", 0.0))
        self.init_mode = generation_cfg.get("init_mode", "random")

    def generate_nominal_mesh(self, data, meta_data):
        raise Exception("Selected generator does not generate a nominal mesh.")

    def generate_latent(self, data):
        wrist_wrench = torch.from_numpy(data["wrist_wrench"]).float().unsqueeze(0).to(self.device)
        partial_pointcloud = torch.from_numpy(data["partial_pointcloud"]).float().unsqueeze(0).to(self.device)

        z_deform = self.model.encode_pointcloud(partial_pointcloud)
        z_wrench = self.model.encode_wrench(wrist_wrench)
        return (z_deform, z_wrench), {}

    def generate_mesh(self, data, meta_data):
        # Check if we have been provided with the latent already.
        if "latent" in meta_data:
            latent = meta_data["latent"]
        else:
            # Generate deformation code latent.
            latent, _ = self.generate_latent(data)
        z_deform, z_wrench = latent

        # Setup function to map from query points to SDF values.
        def sdf_fn(query_points):
            return self.model.forward(query_points.unsqueeze(0), z_deform, z_wrench)["sdf_dist"].mean[0]

        start = time.time()
        mesh = create_mesh(sdf_fn, n=self.mesh_resolution)
        end = time.time()
        mesh_gen_time = end - start

        return mesh, {"latent": latent, "mesh": mesh, "mesh_gen_time": mesh_gen_time}

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
        z_deform, z_wrench = latent

        # We sample contact patch by sampling points on the surface of the mesh and finding any on contact until desired
        # number is met.
        contact_patch = []
        num_contacts_found = 0

        iters = 0
        while num_contacts_found < self.contact_patch_size:
            iters += 1
            surface_point_samples_np = mesh.sample(100000)
            surface_point_samples = torch.from_numpy(surface_point_samples_np).float().to(self.device).unsqueeze(0)
            surface_pred_dict = self.model.forward(surface_point_samples, z_deform, z_wrench)
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
        raise NotImplementedError
        # Check if we have been provided with the latent already.
        if "latent" in meta_data:
            latent = meta_data["latent"]
        else:
            # Generate deformation code latent.
            latent, _ = self.generate_latent(data)

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

    def generate_iou_labels(self, data, metadata):
        raise NotImplementedError
        # Check if we have been provided with the latent already.
        if "latent" in metadata:
            latent = metadata["latent"]
        else:
            # Generate deformation code latent.
            latent, _ = self.generate_latent(data)

        object_idx = torch.from_numpy(data["object_idx"]).to(self.device)
        wrist_wrench = torch.from_numpy(data["wrist_wrench"]).to(self.device).float().unsqueeze(0)
        z_object = self.model.encode_object(object_idx)
        z_wrench = self.model.encode_wrench(wrist_wrench)

        # Get points to evaluate iou.
        points_iou = torch.from_numpy(data["points_iou"]).to(self.device).float().unsqueeze(0)

        # Get occupancy predictions from SDF predictions.
        iou_pred_dict = self.model.forward(points_iou, latent, z_object, z_wrench)
        sdf_pred = iou_pred_dict["sdf"].squeeze(0)
        iou_labels = sdf_pred < 0.0

        return {"iou_labels": iou_labels, "iou_sdf": sdf_pred}, {"latent": latent}
