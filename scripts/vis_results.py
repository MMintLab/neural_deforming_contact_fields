import argparse
import os

import mmint_utils
import numpy as np
import trimesh
from neural_contact_fields.utils import mesh_utils, vedo_utils
from neural_contact_fields.utils.model_utils import load_dataset_from_config
from neural_contact_fields.utils.results_utils import load_gt_results, load_pred_results
from vedo import Plotter, Mesh, Points, LegendBox


def vis_mesh_prediction(partial_pointcloud: np.ndarray,
                        pred_mesh: trimesh.Trimesh, gt_mesh: trimesh.Trimesh,
                        pred_pointcloud: np.ndarray, gt_pointcloud: np.ndarray,
                        pred_contact_patch: np.ndarray, gt_contact_patch: np.ndarray,
                        surface_points: np.ndarray, pred_surface_contact: np.ndarray, gt_surface_contact: np.ndarray,
                        ):
    plt = Plotter(shape=(2, 3))

    # First show the input pointcloud used.
    plt.at(0).show(vedo_utils.draw_origin(), Points(partial_pointcloud), "Partial PC (Input)")

    # Show the ground truth geometry/contact patch.
    gt_mesh_vedo = Mesh([gt_mesh.vertices, gt_mesh.faces])
    _, gt_contact_triangles, _ = mesh_utils.find_in_contact_triangles(gt_mesh, surface_points[gt_surface_contact])
    gt_colors = [[255, 0, 0, 255] if c else [255, 255, 0, 255] for c in gt_contact_triangles]
    gt_mesh_vedo.celldata["CellIndividualColors"] = np.array(gt_colors).astype(np.uint8)
    gt_mesh_vedo.celldata.select("CellIndividualColors")
    plt.at(3).show(gt_mesh_vedo, vedo_utils.draw_origin(), "GT Mesh/Contact")

    # Show predicted mesh, if provided.
    if pred_mesh is not None:
        pred_geom_mesh = Mesh([pred_mesh.vertices, pred_mesh.faces])
        plt.at(1).show(pred_geom_mesh, vedo_utils.draw_origin(), "Pred. Mesh")

    # Show predicted pointcloud, if provided.
    if pred_pointcloud is not None:
        pred_geom_pc = Points(pred_pointcloud)
        plt.at(4).show(pred_geom_pc, vedo_utils.draw_origin(), "Pred. PC")

    if pred_contact_patch is not None:
        pred_patch_pc = Points(pred_contact_patch, c="red").legend("Predicted")
        gt_patch_pc = Points(gt_contact_patch, c="blue").legend("Ground Truth")
        leg = LegendBox([pred_patch_pc, gt_patch_pc])
        plt.at(2).show(pred_patch_pc, gt_patch_pc, leg, vedo_utils.draw_origin(), "Pred. Contact Patch")

    # Show predicted contact labels, if provided.
    if pred_surface_contact is not None:
        _, pred_contact_triangles, _ = mesh_utils.find_in_contact_triangles(gt_mesh,
                                                                            surface_points[
                                                                                pred_surface_contact.cpu().numpy()])
        pred_surface_mesh = Mesh([gt_mesh.vertices, gt_mesh.faces])
        pred_colors = [[255, 0, 0, 255] if c else [255, 255, 0, 255] for c in pred_contact_triangles]
        pred_surface_mesh.celldata["CellIndividualColors"] = np.array(pred_colors).astype(np.uint8)
        pred_surface_mesh.celldata.select("CellIndividualColors")
        plt.at(5).show(pred_surface_mesh, vedo_utils.draw_origin(), "Pred. Contact")

    plt.interactive().close()


def vis_results(dataset_cfg: str, gen_dir: str, mode: str = "test", partial: bool = False):
    # Load dataset.
    dataset_cfg, dataset = load_dataset_from_config(dataset_cfg, dataset_mode=mode)
    num_trials = len(dataset)

    # Load specific ground truth results needed for evaluation.
    gt_meshes, gt_pointclouds, gt_contact_patches, gt_contact_labels, points_iou, gt_occ_iou = load_gt_results(
        dataset, dataset_cfg["data"][mode]["dataset_dir"], num_trials
    )

    # Load predicted results.
    pred_meshes, pred_pointclouds, pred_contact_patches, pred_contact_labels = load_pred_results(gen_dir, num_trials)

    for trial_idx in range(len(dataset)):
        trial_dict = dataset[trial_idx]

        # Load the conditioning pointcloud used.
        if partial:
            pc = trial_dict["partial_pointcloud"]
        else:
            pc = trial_dict["surface_points"]

        # Load surface predictions.
        if pred_contact_labels[trial_idx] is not None:
            surface_pred_dict = pred_contact_labels[trial_idx]
            surface_label = surface_pred_dict["contact_labels"]
        else:
            surface_label = None

        vis_mesh_prediction(pc,
                            pred_meshes[trial_idx], gt_meshes[trial_idx],
                            pred_pointclouds[trial_idx], gt_pointclouds[trial_idx],
                            pred_contact_patches[trial_idx], gt_contact_patches[trial_idx],
                            trial_dict["surface_points"], surface_label, trial_dict["surface_in_contact"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize generated results.")
    parser.add_argument("dataset_config", type=str, help="Dataset config.")
    parser.add_argument("gen_dir", type=str, help="Generation directory.")
    parser.add_argument("--mode", "-m", type=str, default="test", help="Dataset mode [train, val, test].")
    parser.add_argument("--partial", "-p", dest="partial", action='store_true', help='Generated with partial pc.')
    parser.set_defaults(partial=False)
    args = parser.parse_args()

    vis_results(args.dataset_config, args.gen_dir, args.mode, args.partial)
