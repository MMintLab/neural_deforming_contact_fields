import argparse
import os
import pdb
import random

import mmint_utils
import numpy as np
import pytorch3d.loss
import torch
import trimesh
from neural_contact_fields.utils import mesh_utils, vedo_utils, utils
from neural_contact_fields.utils.model_utils import load_dataset_from_config
from neural_contact_fields.utils.results_utils import load_gt_results, load_pred_results, load_gt_results_real
from vedo import Plotter, Mesh, Points, LegendBox


def vis_mesh_prediction_real(partial_pointcloud: np.ndarray,
                             pred_mesh: trimesh.Trimesh,
                             pred_pointcloud: np.ndarray,
                             pred_contact_patch: np.ndarray,
                             gt_contact_patch: np.ndarray,
                             ):
    plt = Plotter(shape=(1, 3))

    # First show the input pointcloud used.
    plt.at(0).show(vedo_utils.draw_origin(), Points(partial_pointcloud), "Partial PC (Input)")

    # Show predicted mesh, if provided.
    if pred_mesh is not None:
        pred_patch_pc = Points(pred_contact_patch, c="red").legend("Predicted")
        pred_geom_mesh = Mesh([pred_mesh.vertices, pred_mesh.faces])
        plt.at(1).show(pred_geom_mesh, vedo_utils.draw_origin(), "Pred. Mesh", pred_patch_pc)

    # Show predicted pointcloud, if provided.
    if pred_pointcloud is not None:
        pred_geom_pc = Points(pred_pointcloud)
        plt.at(1).show(pred_geom_pc, vedo_utils.draw_origin(), "Pred. PC")

    if pred_contact_patch is not None:
        pred_patch_pc = Points(pred_contact_patch, c="red").legend("Predicted")
        gt_patch_pc = Points(gt_contact_patch, c="blue").legend("Ground Truth")
        leg = LegendBox([pred_patch_pc, gt_patch_pc])
        plt.at(2).show(pred_patch_pc, gt_patch_pc, leg, vedo_utils.draw_origin(), "Pred. Contact Patch")

    plt.interactive().close()


def vis_results(dataset_cfg: str, gen_dir: str, mode: str = "test"):
    # Load dataset.
    dataset_cfg, dataset = load_dataset_from_config(dataset_cfg, dataset_mode=mode)
    num_trials = len(dataset)

    # Load specific ground truth results needed for evaluation.
    gt_contact_patches = load_gt_results_real(
        dataset, dataset_cfg["data"][mode]["dataset_dir"], num_trials
    )

    # Load predicted results.
    pred_meshes, pred_pointclouds, pred_contact_patches, pred_contact_labels = load_pred_results(gen_dir, num_trials)

    for trial_idx in range(len(dataset)):
        trial_dict = dataset[trial_idx]

        # Load the conditioning pointcloud used.
        pc = trial_dict["partial_pointcloud"]

        pred_pc = pred_contact_patches[trial_idx]
        gt_pc = utils.sample_pointcloud(gt_contact_patches[trial_idx], 300)

        patch_chamfer_dist, _ = pytorch3d.loss.chamfer_distance(
            pred_pc.unsqueeze(0).float(),
            gt_pc.unsqueeze(0).float())
        print(patch_chamfer_dist.item() * 1e6)

        vis_mesh_prediction_real(pc, pred_meshes[trial_idx], pred_pointclouds[trial_idx],
                                 pred_contact_patches[trial_idx], gt_contact_patches[trial_idx])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize generated results.")
    parser.add_argument("dataset_config", type=str, help="Dataset config.")
    parser.add_argument("gen_dir", type=str, help="Generation directory.")
    parser.add_argument("--mode", "-m", type=str, default="test", help="Dataset mode [train, val, test].")
    args = parser.parse_args()

    torch.manual_seed(10)
    np.random.seed(10)
    random.seed(10)

    vis_results(args.dataset_config, args.gen_dir, args.mode)
