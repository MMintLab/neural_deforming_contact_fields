import torch
import trimesh
import numpy as np
from neural_contact_fields import metrics as ncf_metrics
from neural_contact_fields.utils.mesh_utils import occupancy_check


def sample_points_in_bounds(mesh: trimesh.Trimesh, n_random: int, bound_extend: float = 0.03):
    bounds = mesh.bounds
    min_bounds = bounds[0] - bound_extend
    max_bounds = bounds[1] + bound_extend
    query_points_random = min_bounds + (np.random.random((n_random, 3)) * (max_bounds - min_bounds))
    return query_points_random


device = torch.device("cuda:0")

mesh_a = trimesh.load("out_0_mesh.obj")
mesh_b = trimesh.load("mesh_0.obj")

n = 10

# Effect of num of points.
for n_1 in [10000, 100000, 1000000]:
    ious = []
    for run in range(n):
        points_1 = sample_points_in_bounds(mesh_a, n_1, bound_extend=0.01)
        tgt_1 = occupancy_check(mesh_a, points_1)
        iou_1 = ncf_metrics.mesh_iou(torch.from_numpy(points_1).to(device), torch.from_numpy(tgt_1).to(device), mesh_b,
                                     device)
        ious.append(iou_1.item())
    print("IOU: n=%d, iou=%f (%f)" % (n_1, np.mean(ious), np.std(ious)))

# Effect of num of points.
for b_1 in [0.01, 0.03, 0.05]:
    ious = []
    for run in range(n):
        points_1 = sample_points_in_bounds(mesh_a, 1000000, bound_extend=b_1)
        tgt_1 = occupancy_check(mesh_a, points_1)
        iou_1 = ncf_metrics.mesh_iou(torch.from_numpy(points_1).to(device), torch.from_numpy(tgt_1).to(device), mesh_b,
                                     device)
        ious.append(iou_1.item())
    print("IOU: b=%f, iou=%f (%f)" % (b_1, np.mean(ious), np.std(ious)))
