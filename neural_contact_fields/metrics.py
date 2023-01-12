import torch
import trimesh
import pytorch3d.loss


# TODO: Unit test.
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

    precision = tp.sum() / (tp.sum() + fp.sum())
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


def mesh_chamfer_distance(pred_mesh: trimesh.Trimesh, gt_mesh: trimesh.Trimesh, n: int = 100000):
    """
    Calculate chamfer distance between predicted and ground truth mesh.

    Args:
    - pred_mesh (trimesh.Trimesh): predicted mesh
    - gt_mesh (trimesh.Trimesh): ground truth mesh
    - n (int): number of samples to take on surface
    """
    pred_pc = pred_mesh.sample(n)
    gt_pc = gt_mesh.sample(n)

    chamfer_dist = pytorch3d.loss.chamfer_distance(torch.from_numpy(pred_pc), torch.from_numpy(gt_pc))
    return chamfer_dist
