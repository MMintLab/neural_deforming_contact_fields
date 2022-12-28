import torch
import torch.nn.functional as F
import neural_contact_fields.utils.diff_operators as diff_operators
from pytorch3d.loss import chamfer_distance


# TODO: Unit tests for losses.


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


def surface_normal_loss(gt_sdf: torch.Tensor, gt_normals: torch.Tensor, pred_sdf: torch.Tensor, coords: torch.Tensor):
    """
    Surface normal loss. Encourage the surface normals predicted to match the ground truth.

    Args:
    - gt_sdf (torch.Tensor): ground truth sdf values.
    - gt_normals (torch.Tensor): ground truth surface normals.
    - pred_sdf (torch.Tensor): predicted sdf values.
    - coords (torch.Tensor): coordinates for normals/predicted.
    """
    # Calculate the predicted gradient from the implicit model.
    pred_normals = diff_operators.gradient(pred_sdf, coords)

    # Normalize predicted normals to be length 1.0
    pred_normals_unit = pred_normals / torch.linalg.norm(pred_normals, dim=-1)[..., None]

    # Calculate difference between predicted and gt normals.
    diff = (1.0 - F.cosine_similarity(pred_normals_unit, gt_normals, dim=-1))
    diff_loss = torch.where(gt_sdf == 0.0, diff, torch.tensor(0.0, dtype=pred_sdf.dtype, device=pred_sdf.device))

    # Average error (for surface points).
    norm_loss = diff_loss.sum() / (gt_sdf == 0.0).sum()
    return norm_loss


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
    nominal_surface_points = nominal_coords[nominal_sdf == 0.0]
    predicted_surface_points = pred_def_coords[gt_sdf == 0.0]

    # Calculate the chamfer distance between the extracted surfaces.
    c_loss = chamfer_distance(nominal_surface_points, predicted_surface_points)

    return c_loss.mean()


def deform_loss(pred_def: torch.Tensor):
    """
    Deformation loss. L2 of predicted deformation. Encourages smaller deformations (to smooth deformation fields).

    Args:
    - pred_def (torch.Tensor): predicted deformation field.
    """
    pred_def_norm = torch.linalg.vector_norm(pred_def, dim=-1)

    return pred_def_norm.mean()


def hypo_weight_loss(hypo_params: torch.Tensor):
    """
    Hypo Weight Loss. L2 Squared of predicted hypernetwork parameters.

    Args:
    - hypo_params (torch.Tensor): predicted parameters.
    """
    weight_sum = 0.0
    total_weights = 0

    for weight in hypo_params.values():
        weight_sum += torch.sum(weight ** 2.0)
        total_weights += weight.numel()

    return weight_sum * (1.0 / total_weights)


def embedding_loss(embeddings: torch.Tensor):
    """
    Embedding loss. L2 Squared of predicted embeddings.
    """
    embedding_sizes = torch.sum(embeddings ** 2, dim=-1)
    return torch.mean(embedding_sizes)
