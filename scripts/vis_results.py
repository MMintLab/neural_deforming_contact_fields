import argparse
import os

import mmint_utils
import numpy as np
import trimesh
from neural_contact_fields.utils import mesh_utils, vedo_utils
from neural_contact_fields.utils.model_utils import load_dataset_from_config
from vedo import Plotter, Mesh, Points


def vis_mesh_prediction(partial_pointcloud: np.ndarray,
                        pred_mesh: trimesh.Trimesh, gt_mesh: trimesh.Trimesh, pred_surface_contact: np.ndarray,
                        gt_surface_contact: np.ndarray):
    plt = Plotter(shape=(2, 2))

    _, gt_contact_triangles, _ = mesh_utils.find_in_contact_triangles(gt_mesh, gt_surface_contact)
    _, pred_contact_triangles, _ = mesh_utils.find_in_contact_triangles(gt_mesh, pred_surface_contact)

    plt.at(0).show(vedo_utils.draw_origin(), Points(partial_pointcloud), "Partial PC")

    gt_mesh_vedo = Mesh([gt_mesh.vertices, gt_mesh.faces])
    gt_colors = [[255, 0, 0, 255] if c else [255, 255, 0, 255] for c in gt_contact_triangles]
    gt_mesh_vedo.celldata["CellIndividualColors"] = np.array(gt_colors).astype(np.uint8)
    gt_mesh_vedo.celldata.select("CellIndividualColors")
    plt.at(1).show(gt_mesh_vedo, vedo_utils.draw_origin(), "GT Mesh")

    pred_surface_mesh = Mesh([gt_mesh.vertices, gt_mesh.faces])
    pred_colors = [[255, 0, 0, 255] if c else [255, 255, 0, 255] for c in pred_contact_triangles]
    pred_surface_mesh.celldata["CellIndividualColors"] = np.array(pred_colors).astype(np.uint8)
    pred_surface_mesh.celldata.select("CellIndividualColors")
    plt.at(2).show(pred_surface_mesh, vedo_utils.draw_origin(), "Pred. Contact")

    pred_geom_mesh = Mesh([pred_mesh.vertices, pred_mesh.faces])
    plt.at(3).show(pred_geom_mesh, vedo_utils.draw_origin(), "Pred. Mesh")

    plt.interactive().close()


def vis_results(dataset_cfg: str, gen_dir: str, mode: str = "test", partial: bool = False):
    # Load dataset.
    dataset_cfg, dataset = load_dataset_from_config(dataset_cfg, dataset_mode=mode)

    # Load ground truth meshes. # TODO: Consider case where GT mesh not available.
    gt_meshes = []
    dataset_dir = dataset_cfg["data"][mode]["dataset_dir"]
    for trial_idx in range(len(dataset)):
        mesh_fn = os.path.join(dataset_dir, "out_%d_mesh.obj" % trial_idx)
        mesh = trimesh.load(mesh_fn)
        gt_meshes.append(mesh)

    # Load predicted data.
    gen_dir = gen_dir
    pred_meshes = []
    surface_pred_dicts = []
    for trial_idx in range(len(dataset)):
        pred_mesh_fn = os.path.join(gen_dir, "mesh_%d.obj" % trial_idx)
        if os.path.exists(pred_mesh_fn):
            pred_mesh = trimesh.load(pred_mesh_fn)
        else:
            pred_mesh = None
        pred_meshes.append(pred_mesh)

        surface_pred_fn = os.path.join(gen_dir, "contact_labels_%d.pkl.gzip" % trial_idx)
        surface_pred_dict = mmint_utils.load_gzip_pickle(surface_pred_fn)
        surface_pred_dicts.append(surface_pred_dict)

    for trial_idx in range(len(dataset)):
        trial_dict = dataset[trial_idx]
        surface_pred_dict = surface_pred_dicts[trial_idx]

        if partial:
            pc = trial_dict["partial_pointcloud"]
        else:
            pc = trial_dict["surface_points"]

        vis_mesh_prediction(pc, pred_meshes[trial_idx], gt_meshes[trial_idx],
                            trial_dict["surface_points"][surface_pred_dict.cpu() > 0.5],
                            trial_dict["surface_points"][trial_dict["surface_in_contact"]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize generated results.")
    parser.add_argument("dataset_config", type=str, help="Dataset config.")
    parser.add_argument("gen_dir", type=str, help="Generation directory.")
    parser.add_argument("--mode", "-m", type=str, default="test", help="Dataset mode [train, val, test].")
    parser.add_argument("--partial", "-p", dest="partial", action='store_true', help='Generated with partial pc.')
    parser.set_defaults(partial=False)
    args = parser.parse_args()

    vis_results(args.dataset_config, args.gen_dir, args.mode, args.partial)
