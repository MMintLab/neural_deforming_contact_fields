import torch
import trimesh
import pytorch3d.loss
from neural_contact_fields.utils.mesh_utils import occupancy_check


def precision_recall(pred: torch.Tensor, gt: torch.Tensor):
    """
    For a set of binary predictions/GT, determine the precision recall
    statistics.

    Args:
    - pred (torch.Tensor): binary predictions
    - gt (torch.Tensor): binary ground truth
    """
    assert pred.shape == gt.shape
    assert pred.dtype == torch.bool and gt.dtype == torch.bool

    tp = torch.logical_and(pred, gt)
    fp = torch.logical_and(pred, torch.logical_not(gt))
    fn = torch.logical_and(torch.logical_not(pred), gt)
    tn = torch.logical_and(torch.logical_not(pred), torch.logical_not(gt))

    if tp.sum() + fp.sum() == 0:
        precision = 0.0
    else:
        precision = tp.sum() / (tp.sum() + fp.sum())

    if tp.sum() + fn.sum() == 0:
        recall = 0.0
    else:
        recall = tp.sum() / (tp.sum() + fn.sum())

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
    }


def binary_accuracy(pred: torch.Tensor, gt: torch.Tensor):
    """
    For a set of binary predictions/GT, determine the binary accuracy.

    Args:
    - pred (torch.Tensor): binary predictions
    - gt (torch.Tensor): binary ground truth
    """
    assert pred.shape == gt.shape
    assert pred.dtype == torch.bool and gt.dtype == torch.bool

    return (pred == gt).sum(-1) / pred.shape[-1]


def mesh_chamfer_distance(pred_mesh: trimesh.Trimesh, gt_mesh: trimesh.Trimesh, n: int = 10000, device=None):
    """
    Calculate chamfer distance between predicted and ground truth mesh.

    Args:
    - pred_mesh (trimesh.Trimesh): predicted mesh
    - gt_mesh (trimesh.Trimesh): ground truth mesh
    - n (int): number of samples to take on surface
    """
    pred_pc = pred_mesh.sample(n)
    gt_pc = gt_mesh.sample(n)

    chamfer_dist = pytorch3d.loss.chamfer_distance(torch.from_numpy(pred_pc).unsqueeze(0).float().to(device),
                                                   torch.from_numpy(gt_pc).unsqueeze(0).float().to(device))
    return chamfer_dist


def intersection_over_union(pred: torch.Tensor, gt: torch.Tensor):
    """
    Calculate Intersection over Union.

    Args:
    - pred (torch.Tensor): predicted occ
    - gt (torch.Tensor): gt occ
    """
    assert pred.shape == gt.shape
    assert pred.dtype == torch.bool and gt.dtype == torch.bool

    union = torch.logical_or(pred, gt).float().sum(axis=-1)
    intersect = torch.logical_and(pred, gt).float().sum(axis=-1)

    iou = intersect / union
    return iou


def mesh_iou(points_iou: torch.Tensor, occ_tgt: torch.Tensor, pred_mesh: trimesh.Trimesh, device=None):
    """
    Calculate mesh intersection over union. Given target points and their ground truth occupancy, determine whether
    the target points are occupied in the predicted mesh and calculated IoU accordingly.

    Args:
    - points_iou (torch.Tensor): target points in space near object
    - occ_tgt (torch.Tensor): ground truth occupancy of target points
    - pred_mesh (trimesh.Trimesh):
    """
    # Check occupancy of target points.
    query_points = points_iou.cpu().numpy()
    occ_pred = occupancy_check(pred_mesh, query_points)
    occ_pred = torch.from_numpy(occ_pred).to(device)

    iou = intersection_over_union(occ_pred, occ_tgt)

    return iou