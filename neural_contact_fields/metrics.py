import torch


def precision_recall(pred: torch.Tensor, gt: torch.Tensor):
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
