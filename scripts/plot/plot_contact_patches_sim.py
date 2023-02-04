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
base_test_dir = "out/experiments/terrain_tests_v2/partial_pointcloud_vis/"

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

pred_patches_all = []
pred_meshes_all = []
metrics_all = []
for title, test_dir in zip(titles, test_dirs):
    gen_dir = os.path.join(base_test_dir, test_dir)
    # Load predicted results.
    pred_meshes, pred_pointclouds, pred_contact_patches, pred_contact_labels = load_pred_results(gen_dir, num_trials)

    pred_patches_all.append(pred_contact_patches)
    pred_meshes_all.append(pred_meshes)

# Build plots.
for index in trange(num_trials):
    trial_dict = dataset[index]
    # pc = trial_dict["partial_pointcloud"]

    plt = Plotter(shape=(1, 3))

    for method_idx in range(len(titles)):
        pred_patch = pred_patches_all[method_idx][index]
        plt.at(method_idx).show(Points(pred_patch, c="red"))

    plt.at(len(titles)).show(Points(gt_contact_patches[index], c="blue"))
    plt.interactive().close()
