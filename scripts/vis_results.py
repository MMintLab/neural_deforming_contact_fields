import argparse

from tqdm import trange
import numpy as np
from neural_contact_fields.utils import mesh_utils, vedo_utils
from neural_contact_fields.utils.model_utils import load_dataset_from_config
from neural_contact_fields.utils.results_utils import load_gt_results, load_pred_results
from vedo import Plotter, Mesh, Points, LegendBox


def vis_mesh_prediction(data_dict: dict, gen_dict: dict, gt_dict: dict):
    plt = Plotter(shape=(2, 3))

    # First show the input pointcloud used.
    plt.at(0).show(vedo_utils.draw_origin(), Points(data_dict["partial_pointcloud"]), "Partial PC (Input)")

    # Show the ground truth geometry/contact patch.
    gt_mesh = gt_dict["mesh"]
    surface_points = data_dict["surface_points"]
    gt_surface_contact = data_dict["surface_in_contact"]
    gt_mesh_vedo = Mesh([gt_mesh.vertices, gt_mesh.faces])
    _, gt_contact_triangles, _ = mesh_utils.find_in_contact_triangles(gt_mesh, surface_points[gt_surface_contact])
    gt_colors = [[255, 0, 0, 255] if c else [255, 255, 0, 255] for c in gt_contact_triangles]
    gt_mesh_vedo.celldata["CellIndividualColors"] = np.array(gt_colors).astype(np.uint8)
    gt_mesh_vedo.celldata.select("CellIndividualColors")
    plt.at(3).show(gt_mesh_vedo, vedo_utils.draw_origin(), "GT Mesh/Contact")

    # Show predicted mesh, if provided.
    if "mesh" in gen_dict:
        pred_mesh = gen_dict["mesh"]
        pred_geom_mesh = Mesh([pred_mesh.vertices, pred_mesh.faces])
        plt.at(1).show(pred_geom_mesh, vedo_utils.draw_origin(), "Pred. Mesh")

    # Show predicted pointcloud, if provided.
    if "pointcloud" in gen_dict and gen_dict["pointcloud"] is not None:
        pred_pointcloud = gen_dict["pointcloud"]
        pred_geom_pc = Points(pred_pointcloud)
        plt.at(4).show(pred_geom_pc, vedo_utils.draw_origin(), "Pred. PC")

    if "contact_patch" in gen_dict and gen_dict["contact_patch"] is not None:
        pred_contact_patch = gen_dict["contact_patch"]
        gt_contact_patch = gt_dict["contact_patch"]
        pred_patch_pc = Points(pred_contact_patch, c="red").legend("Predicted")
        gt_patch_pc = Points(gt_contact_patch, c="blue").legend("Ground Truth")
        leg = LegendBox([pred_patch_pc, gt_patch_pc])
        plt.at(2).show(pred_patch_pc, gt_patch_pc, leg, vedo_utils.draw_origin(), "Pred. Contact Patch")

    # Show predicted contact labels, if provided.
    if "contact_labels" in gen_dict and gen_dict["contact_labels"] is not None:
        pred_surface_contact = gen_dict["contact_labels"]
        _, pred_contact_triangles, _ = mesh_utils.find_in_contact_triangles(gt_mesh,
                                                                            surface_points[
                                                                                pred_surface_contact.cpu().numpy()])
        pred_surface_mesh = Mesh([gt_mesh.vertices, gt_mesh.faces])
        pred_colors = [[255, 0, 0, 255] if c else [255, 255, 0, 255] for c in pred_contact_triangles]
        pred_surface_mesh.celldata["CellIndividualColors"] = np.array(pred_colors).astype(np.uint8)
        pred_surface_mesh.celldata.select("CellIndividualColors")
        plt.at(5).show(pred_surface_mesh, vedo_utils.draw_origin(), "Pred. Contact")

    plt.interactive().close()


def vis_results(dataset_cfg: str, gen_dir: str, mode: str = "test", offset: int = 0):
    # Load dataset.
    dataset_cfg, dataset = load_dataset_from_config(dataset_cfg, dataset_mode=mode)
    num_trials = len(dataset)

    # Load specific ground truth results needed for evaluation.
    gt_dicts = load_gt_results(dataset, num_trials)

    # Load predicted results.
    gen_dicts = load_pred_results(gen_dir, num_trials)

    for trial_idx in trange(offset, len(dataset)):
        trial_dict = dataset[trial_idx]

        vis_mesh_prediction(trial_dict, gen_dicts[trial_idx], gt_dicts[trial_idx])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize generated results.")
    parser.add_argument("dataset_config", type=str, help="Dataset config.")
    parser.add_argument("gen_dir", type=str, help="Generation directory.")
    parser.add_argument("--mode", "-m", type=str, default="test", help="Dataset mode [train, val, test].")
    parser.add_argument("--offset", "-o", type=int, default=0, help="Offset to start from.")
    args = parser.parse_args()

    vis_results(args.dataset_config, args.gen_dir, args.mode, args.offset)
