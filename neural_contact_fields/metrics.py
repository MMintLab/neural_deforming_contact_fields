import torch
import torchmetrics
import trimesh
try:
    import pytorch3d.loss
except:
    print("pytorch3d not installed ")
from neural_contact_fields.utils import vedo_utils
from neural_contact_fields.utils.mesh_utils import occupancy_check
from vedo import Plotter, Points


def mesh_chamfer_distance(pred_mesh: trimesh.Trimesh, gt_mesh: trimesh.Trimesh, n: int = 10000, device=None,
                          vis: bool = False):
    """
    Calculate chamfer distance between predicted and ground truth mesh.

    Args:
    - pred_mesh (trimesh.Trimesh): predicted mesh
    - gt_mesh (trimesh.Trimesh): ground truth mesh
    - n (int): number of samples to take on surface
    """
    pred_pc = pred_mesh.sample(n)
    gt_pc = gt_mesh.sample(n)

    if vis:
        plt = Plotter(shape=(1, 2))
        plt.at(0).show(Points(gt_pc), vedo_utils.draw_origin(), "GT")
        plt.at(1).show(Points(pred_pc), vedo_utils.draw_origin(), "Pred.")
        plt.interactive()

    chamfer_dist, _ = pytorch3d.loss.chamfer_distance(torch.from_numpy(pred_pc).unsqueeze(0).float().to(device),
                                                      torch.from_numpy(gt_pc).unsqueeze(0).float().to(device))
    return chamfer_dist


def mesh_iou(points_iou: torch.Tensor, occ_tgt: torch.Tensor, pred_mesh: trimesh.Trimesh, device=None,
             vis: bool = False):
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
    occ_pred = torch.from_numpy(occ_pred).to(device).float()

    if vis:
        plt = Plotter(shape=(1, 3))
        plt.at(0).show(Points(query_points), vedo_utils.draw_origin(), "Sample Points")
        plt.at(1).show(Points(query_points[occ_tgt.bool().cpu().numpy()]), vedo_utils.draw_origin(), "GT Occ.")
        plt.at(2).show(Points(query_points[occ_pred.bool().cpu().numpy()]), vedo_utils.draw_origin(), "Pred Occ.")
        plt.interactive()

    iou = torchmetrics.functional.classification.binary_jaccard_index(occ_pred, occ_tgt, threshold=0.5)

    return iou