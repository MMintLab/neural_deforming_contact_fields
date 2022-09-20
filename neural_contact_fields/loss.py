import torch
import torch.nn.functional as F


def sdf_loss_clamp(sdf_pred: torch.Tensor, sdf_gt: torch.Tensor, clamp: float, reduce: str = "mean"):
    """
    SDF Loss clamped.
    """
    sdf_pred_clamp = torch.clamp(sdf_pred, max=clamp)
    sdf_gt_clamp = torch.clamp(sdf_gt, max=clamp)

    sdf_err = torch.abs(sdf_pred_clamp - sdf_gt_clamp)

    if reduce == "mean":
        return sdf_err.mean()
    elif reduce == "sum":
        return sdf_err.sum()
    elif reduce == "none":
        return sdf_err
    else:
        raise Exception("Unknown reduction: %s." % reduce)
