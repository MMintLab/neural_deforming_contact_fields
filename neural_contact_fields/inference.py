from collections import defaultdict

import numpy as np
import torch
import tqdm
from neural_contact_fields.neural_contact_field.models.neural_contact_field import NeuralContactField
from neural_contact_fields.utils.infer_utils import inference_by_optimization
from neural_contact_fields.utils.marching_cubes import create_mesh
from torch import nn
import torch.optim as optim
import neural_contact_fields.loss as ncf_losses
import torch.nn.functional as F


def infer_latent(model: NeuralContactField, trial_dict: dict, loss_weights: dict, device=None):
    z_deform_size = model.z_deform_size

    def infer_loss_fn(model_, latent_, data_dict):
        # Pull out relevant data.
        object_idx_ = torch.from_numpy(data_dict["object_idx"]).to(device)
        coords_ = torch.from_numpy(data_dict["query_point"]).to(device).float().unsqueeze(0)
        trial_idx = torch.from_numpy(data_dict["trial_idx"]).to(device)
        gt_sdf = torch.from_numpy(data_dict["sdf"]).to(device).float().unsqueeze(0)
        gt_normals = torch.from_numpy(data_dict["normals"]).to(device).float().unsqueeze(0)
        gt_in_contact = torch.from_numpy(data_dict["in_contact"]).to(device).float().unsqueeze(0)
        nominal_coords = torch.from_numpy(data_dict["nominal_query_point"]).to(device).float().unsqueeze(0)
        nominal_sdf = torch.from_numpy(data_dict["nominal_sdf"]).to(device).float().unsqueeze(0)
        wrist_wrench_ = torch.from_numpy(data_dict["wrist_wrench"]).to(device).float().unsqueeze(0)

        # We assume we know the object code.
        z_object_ = model.encode_object(object_idx_)
        z_wrench_ = model.encode_wrench(wrist_wrench_)

        # Predict with updated latents.
        pred_dict_ = model.forward(coords_, latent_, z_object_, z_wrench_)

        # Loss:
        loss_dict = dict()

        # SDF Loss: How accurate are the SDF predictions at each query point.
        sdf_loss = ncf_losses.sdf_loss(pred_dict_["sdf"], gt_sdf)
        loss_dict["sdf_loss"] = sdf_loss

        # Normals loss: are the normals accurate.
        normals_loss = ncf_losses.surface_normal_loss(gt_sdf, gt_normals, pred_dict_["normals"])
        loss_dict["normals_loss"] = normals_loss

        # Latent embedding loss: well-formed embedding.
        embedding_loss = ncf_losses.l2_loss(pred_dict_["embedding"], squared=True)
        loss_dict["embedding_loss"] = embedding_loss

        # Loss on deformation field.
        def_loss = ncf_losses.l2_loss(pred_dict_["deform"], squared=True)
        loss_dict["def_loss"] = def_loss

        # Contact prediction loss.
        contact_loss = F.binary_cross_entropy_with_logits(pred_dict_["in_contact_logits"][gt_sdf == 0.0],
                                                          gt_in_contact[gt_sdf == 0.0])
        loss_dict["contact_loss"] = contact_loss

        # Chamfer distance loss.
        chamfer_loss = ncf_losses.surface_chamfer_loss(nominal_coords, nominal_sdf, gt_sdf, pred_dict_["nominal"])
        loss_dict["chamfer_loss"] = chamfer_loss

        # Calculate total weighted loss.
        loss = 0.0
        for loss_key in loss_dict.keys():
            loss += float(loss_weights[loss_key]) * loss_dict[loss_key]
        loss_dict["loss"] = loss

        return loss

    z_deform_, _ = inference_by_optimization(model, infer_loss_fn, z_deform_size, 1, trial_dict, device=device,
                                             verbose=True)
    z_deform = z_deform_.weight

    # Predict with final latent.
    object_idx = torch.from_numpy(trial_dict["object_idx"]).to(device)
    coords = torch.from_numpy(trial_dict["query_point"]).to(device).float().unsqueeze(0)
    wrist_wrench = torch.from_numpy(trial_dict["wrist_wrench"]).to(device).float().unsqueeze(0)
    z_object = model.encode_object(object_idx)
    z_wrench = model.encode_wrench(wrist_wrench)
    pred_dict = model.forward(coords, z_deform, z_object, z_wrench)

    # Generate mesh.
    latent_sdf_decoder = LatentSDFDecoder(model, z_object, z_deform, z_wrench)
    mesh = create_mesh(latent_sdf_decoder)

    return z_deform_, pred_dict, mesh


def infer_latent_from_surface(model: NeuralContactField, trial_dict: dict, loss_weights: dict, device=None):
    z_deform_size = model.z_deform_size

    def surface_loss_fn(model_, latent, data_dict):
        # Pull out relevant data.
        object_idx_ = torch.from_numpy(data_dict["object_idx"]).to(device)
        surface_coords_ = torch.from_numpy(data_dict["surface_points"]).to(device).float().unsqueeze(0)
        wrist_wrench_ = torch.from_numpy(data_dict["wrist_wrench"]).to(device).float().unsqueeze(0)

        # We assume we know the object code.
        z_object_ = model_.encode_object(object_idx_)
        z_wrench_ = model_.encode_wrench(wrist_wrench_)

        # Predict with updated latents.
        pred_dict_ = model_.forward(surface_coords_, latent, z_object_, z_wrench_)

        # Loss: all points on surface should have SDF = 0.0.
        loss = torch.mean(torch.abs(pred_dict_["sdf"]))

        return loss

    z_deform_, _ = inference_by_optimization(model, surface_loss_fn, z_deform_size, 1, trial_dict, device=device,
                                             verbose=True)
    z_deform = z_deform_.weight

    # Predict with final latent.
    object_idx = torch.from_numpy(trial_dict["object_idx"]).to(device)
    coords = torch.from_numpy(trial_dict["query_point"]).to(device).float().unsqueeze(0)
    wrist_wrench = torch.from_numpy(trial_dict["wrist_wrench"]).to(device).float().unsqueeze(0)
    z_object = model.encode_object(object_idx)
    z_wrench = model.encode_wrench(wrist_wrench)
    pred_dict = model.forward(coords, z_deform, z_object, z_wrench)

    # Predict surface point labels.
    surface_coords = torch.from_numpy(trial_dict["surface_points"]).to(device).float().unsqueeze(0)
    surface_pred_dict = model.forward(surface_coords, z_deform, z_object, z_wrench)

    # Generate mesh.
    latent_sdf_decoder = LatentSDFDecoder(model, z_object, z_deform, z_wrench)
    mesh = create_mesh(latent_sdf_decoder)

    return z_deform_, pred_dict, surface_pred_dict, mesh


def points_inference(model: NeuralContactField, trial_dict, device=None):
    model.eval()
    object_index = trial_dict["object_idx"]
    trial_index = trial_dict["trial_idx"]
    wrist_wrench = torch.from_numpy(trial_dict["wrist_wrench"]).to(device).float().unsqueeze(0)

    # Encode object idx/trial idx.
    z_object, z_trial = model.encode_trial(torch.from_numpy(object_index).to(device),
                                           torch.from_numpy(trial_index).to(device))
    z_wrench = model.encode_wrench(wrist_wrench)

    # Get query points to sample.
    query_points = torch.from_numpy(trial_dict["query_point"]).to(device).float()
    pred_dict = model.forward(query_points.unsqueeze(0), z_trial, z_object, z_wrench)

    return pred_dict


class LatentSDFDecoder(nn.Module):

    def __init__(self, model: NeuralContactField, z_object, z_deform, z_wrench):
        super().__init__()
        self.model: NeuralContactField = model
        self.z_object = z_object
        self.z_deform = z_deform
        self.z_wrench = z_wrench

    def forward(self, query_points: torch.Tensor):
        out_dict = self.model.forward(query_points.unsqueeze(0), self.z_deform, self.z_object, self.z_wrench)
        return out_dict["sdf"].squeeze(0)


def marching_cubes_latent(model: NeuralContactField, trial_dict, device=None):
    model.eval()
    object_index = trial_dict["object_idx"]
    trial_index = trial_dict["trial_idx"]
    wrist_wrench = torch.from_numpy(trial_dict["wrist_wrench"]).to(device).float().unsqueeze(0)

    # Encode object idx/trial idx.
    z_object, z_trial = model.encode_trial(torch.from_numpy(object_index).to(device),
                                           torch.from_numpy(trial_index).to(device))
    z_wrench = model.encode_wrench(wrist_wrench)

    latent_sdf_decoder = LatentSDFDecoder(model, z_object, z_trial, z_wrench)

    return create_mesh(latent_sdf_decoder)
