from collections import OrderedDict

import torch
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance


def sdf_loss(gt_sdf: torch.Tensor, pred_sdf: torch.Tensor, clip: float = 1.0):
    """
    Clipped SDF loss.

    Args:
    - gt_sdf (torch.Tensor): ground truth sdf values.
    - pred_sdf (torch.Tensor): predicted sdf values.
    - clip (float): value to clip predicted/gt sdf values to.
    """
    pred_sdf_clip = torch.clip(pred_sdf, -clip, clip)
    gt_sdf_clip = torch.clip(gt_sdf, -clip, clip)

    loss = torch.abs(pred_sdf_clip - gt_sdf_clip)

    return loss.mean()


def surface_normal_loss(gt_sdf: torch.Tensor, gt_normals: torch.Tensor, pred_normals: torch.Tensor):
    """
    Surface normal loss. Encourage the surface normals predicted to match the ground truth.

    Args:
    - gt_sdf (torch.Tensor): ground truth sdf values.
    - gt_normals (torch.Tensor): ground truth surface normals.
    - pred_sdf (torch.Tensor): predicted sdf values.
    - coords (torch.Tensor): coordinates for normals/predicted.
    """
    # Normalize predicted normals to be length 1.0
    pred_normals_unit = pred_normals / (torch.linalg.norm(pred_normals, dim=-1)[..., None] + 1.0e-8)

    # Calculate difference between predicted and gt normals.
    diff = (1.0 - F.cosine_similarity(pred_normals_unit, gt_normals, dim=-1))
    diff_loss = torch.where(gt_sdf == 0.0, diff, torch.tensor(0.0, dtype=pred_normals.dtype, device=pred_normals.device))

    # Average error (for surface points).
    norm_loss = diff_loss.sum() / (gt_sdf == 0.0).sum()
    return norm_loss


def l2_loss(tens: torch.Tensor, squared: bool = False):
    """
    L2 loss on provided tensor.

    Args:
    - tens (torch.Tensor): tensor to derive l2 loss from.
    - squared (bool): whether to derive squared l2 loss.
    """
    l2_squared = torch.sum(tens ** 2.0, dim=-1)

    if squared:
        return torch.mean(l2_squared)
    else:
        return torch.mean(torch.sqrt(l2_squared))


def surface_chamfer_loss(nominal_coords: torch.Tensor, nominal_sdf: torch.Tensor, gt_sdf: torch.Tensor,
                         pred_def_coords: torch.Tensor):
    """
    Surface chamfer loss. Encourage the deformed surface to match the nominal for the given object.

    Args:
    - nominal_coords (torch.Tensor): nominal object sample coords.
    - nominal_sdf (torch.Tensor): gt nominal object sdf values.
    - gt_sdf (torch.Tensor): gt sdf values (for deformed).
    - pred_def_coords (torch.Tensor): predicted deformed object coords (i.e., predicted nominal).
    """
    # Get the surfaces of the GT nominal and predicted nominal implicits.
    nom_on_surf_idx = torch.where(nominal_sdf == 0)[1]
    def_on_surf_idx = torch.where(gt_sdf == 0)[1]
    nominal_surface_points = nominal_coords[:, nom_on_surf_idx, :]
    predicted_surface_points = pred_def_coords[:, def_on_surf_idx, :]

    # Calculate the chamfer distance between the extracted surfaces.
    c_loss = chamfer_distance(nominal_surface_points, predicted_surface_points)[0]

    return c_loss.mean()


def hypo_weight_loss(hypo_params: OrderedDict):
    """
    Hypo Weight Loss. L2 Squared of predicted hypernetwork parameters.

    Args:
    - hypo_params (OrderedDict): predicted parameters.
    """
    weight_sum = 0.0
    total_weights = 0

    for weight in hypo_params.values():
        weight_sum += torch.sum(weight ** 2.0)
        total_weights += weight.numel()

    return weight_sum * (1.0 / total_weights)
