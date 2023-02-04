import argparse
import os

import mmint_utils
import numpy as np
import trimesh
from neural_contact_fields.utils import mesh_utils, vedo_utils, utils
from neural_contact_fields.utils.model_utils import load_dataset_from_config
from neural_contact_fields.utils.results_utils import load_gt_results, load_pred_results
from tqdm import trange
from vedo import Plotter, Mesh, Points, LegendBox

dataset_cfg = "cfg/dataset/sim_terrain_test_v2.yaml"
mode = "test"
base_test_dir = "out/experiments/terrain_tests_v2/partial_pointcloud"

titles = [
    "Ours",
    "Baseline"
]
test_dirs = [
    "wrench_v2",
    "baseline"
]

# Load dataset.
dataset_cfg, dataset = load_dataset_from_config(dataset_cfg, dataset_mode=mode)
num_trials = len(dataset)

# Load specific ground truth results needed for evaluation.
gt_meshes, gt_pointclouds, gt_contact_patches, gt_contact_labels, points_iou, gt_occ_iou = load_gt_results(
    dataset, dataset_cfg["data"][mode]["dataset_dir"], num_trials
)

pred_meshes_all = []
for title, test_dir in zip(titles, test_dirs):
    gen_dir = os.path.join(base_test_dir, test_dir)
    # Load predicted results.
    pred_meshes, pred_pointclouds, pred_contact_patches, pred_contact_labels = load_pred_results(gen_dir, num_trials)

    pred_meshes_all.append(pred_meshes)

# Build plots.
# for index in trange(num_trials):
for index in [47]:
    trial_dict = dataset[index]
    pc = trial_dict["partial_pointcloud"]

    plt = Plotter(shape=(1, 4))
    plt.at(0).show(Points(pc, c="blue"))

    for method_idx in range(len(titles)):
        pred_mesh = pred_meshes_all[method_idx][index]
        plt.at(1 + method_idx).show(Mesh([pred_mesh.vertices, pred_mesh.faces]))

    plt.at(1 + len(titles)).show(gt_meshes[index])
    plt.interactive().close()
