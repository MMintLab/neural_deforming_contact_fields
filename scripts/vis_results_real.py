import argparse
import random

import numpy as np
import torch
import trimesh
from neural_contact_fields.utils import vedo_utils, utils
from neural_contact_fields.utils.model_utils import load_dataset_from_config
from neural_contact_fields.utils.results_utils import load_pred_results, load_gt_results_real
from tqdm import trange
from vedo import Plotter, Mesh, Points, LegendBox


def vis_inputs(partial_pointcloud: np.ndarray, gt_contact_patch: np.ndarray):
    plt = Plotter(shape=(1, 1))
    plt.at(0).show(Points(partial_pointcloud), Points(gt_contact_patch), vedo_utils.draw_origin())
    plt.interactive().close()


def vis_mesh_prediction_real(partial_pointcloud: np.ndarray,
                             pred_mesh: trimesh.Trimesh,
                             pred_pointcloud: np.ndarray,
                             pred_contact_patch: np.ndarray,
                             gt_contact_patch: np.ndarray,
                             ):
    plt = Plotter(shape=(1, 3))

    # First show the input pointcloud used.
    plt.at(0).show(Points(partial_pointcloud))

    # Show predicted mesh, if provided.
    if pred_mesh is not None:
        pred_patch_pc = Points(pred_contact_patch, c="red").legend("Predicted")
        pred_geom_mesh = Mesh([pred_mesh.vertices, pred_mesh.faces])
        # plt.at(1).show(pred_geom_mesh, vedo_utils.draw_origin(), "Pred. Mesh", pred_patch_pc)
        plt.at(1).show(pred_geom_mesh, Points(partial_pointcloud))

    # Show predicted pointcloud, if provided.
    if pred_pointcloud is not None:
        pred_geom_pc = Points(pred_pointcloud)
        plt.at(1).show(pred_geom_pc, vedo_utils.draw_origin(), "Pred. PC")

    if pred_contact_patch is not None:
        pred_pc = pred_contact_patch
        pred_pc = utils.voxel_downsample_pointcloud(pred_pc, 0.0005)
        pred_pc = utils.sample_pointcloud(pred_pc, 300)
        pred_patch_pc = Points(pred_pc, c="red", alpha=0.3).legend("Predicted")
        gt_patch_pc = Points(gt_contact_patch, c="blue", alpha=0.4).legend("Ground Truth")
        leg = LegendBox([pred_patch_pc, gt_patch_pc])
        # plt.at(2).show(pred_patch_pc, gt_patch_pc, leg, vedo_utils.draw_origin(), "Pred. Contact Patch")
        # plt.at(2).show(pred_patch_pc)
        pred_geom_mesh = Mesh([pred_mesh.vertices, pred_mesh.faces])
        plt.at(2).show(gt_patch_pc, pred_patch_pc, leg)

    plt.interactive().close()


def vis_real_baseline_predictions(pred_contact_patch: np.ndarray, gt_contact_patch: np.ndarray):
    plt = Plotter(shape=(1, 3))

    pred_patch_pc = Points(pred_contact_patch, c="red", alpha=0.1).legend("Predicted")
    gt_patch_pc = Points(gt_contact_patch, c="blue").legend("Ground Truth")
    # leg = LegendBox([pred_patch_pc, gt_patch_pc])
    # plt.at(2).show(pred_patch_pc, gt_patch_pc, leg, vedo_utils.draw_origin(), "Pred. Contact Patch")
    plt.at(0).show(pred_patch_pc)
    plt.at(1).show(gt_patch_pc)

    plt.at(2).show(pred_patch_pc, gt_patch_pc)

    plt.interactive().close()


def vis_results(dataset_cfg: str, gen_dir: str, mode: str = "test", offset: int = 0):
    # Load dataset.
    dataset_cfg, dataset = load_dataset_from_config(dataset_cfg, dataset_mode=mode)
    num_trials = len(dataset)

    # Load specific ground truth results needed for evaluation.
    gt_dicts = load_gt_results_real(dataset, num_trials)

    # Load predicted results.
    gen_dicts = load_pred_results(gen_dir, num_trials)

    for trial_idx in trange(offset, len(dataset)):
        trial_dict = dataset[trial_idx]

        # Load the conditioning pointcloud used.
        pc = trial_dict["partial_pointcloud"]

        vis_mesh_prediction_real(pc, gen_dicts[trial_idx]["mesh"], None,
                                 gen_dicts[trial_idx]["contact_patch"], gt_dicts[trial_idx]["contact_patch"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize generated results.")
    parser.add_argument("dataset_config", type=str, help="Dataset config.")
    parser.add_argument("gen_dir", type=str, help="Generation directory.")
    parser.add_argument("--mode", "-m", type=str, default="test", help="Dataset mode [train, val, test].")
    parser.add_argument("--offset", "-o", type=int, default=0, help="Offset into the dataset.")
    args = parser.parse_args()

    torch.manual_seed(10)
    np.random.seed(10)
    random.seed(10)

    vis_results(args.dataset_config, args.gen_dir, args.mode, args.offset)
