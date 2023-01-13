import argparse
import os
import pdb

import mmint_utils
import numpy as np
import torch
import trimesh
from neural_contact_fields.data.tool_dataset import ToolDataset
from neural_contact_fields.inference import infer_latent_from_surface
from neural_contact_fields.utils import vedo_utils
from neural_contact_fields.utils.model_utils import load_model_and_dataset
from neural_contact_fields.utils import mesh_utils
from scripts.inference.infer_latent import numpy_dict
from scripts.train.vis_prediction_vs_dataset import vis_prediction_vs_dataset
from vedo import Plotter, Mesh
import neural_contact_fields.metrics as ncf_metrics


def get_model_dataset_arg_parser():
    """
    Argument parser for common model + dataset arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Model/data config file.")
    parser.add_argument("--out", "-o", type=str, default=None, help="Directory to write results to.")
    parser.add_argument("--dataset_config", "-d", type=str, default=None, help="Optional dataset config to use.")
    parser.add_argument("--mode", "-m", type=str, default="test", help="Which split to vis [train, val, test].")
    parser.add_argument("--model_file", "-f", type=str, default="model.pt", help="Which model save file to use.")
    parser.add_argument("-v", "--vis", dest="vis", action="store_true", help="Visualize.")
    parser.set_defaults(vis=False)
    return parser


def load_model_dataset_from_args(args):
    """
    Load model and dataset from arguments object.
    """
    model_cfg, model, dataset, device = load_model_and_dataset(args.config, dataset_config=args.dataset_config,
                                                               dataset_mode=args.mode,
                                                               model_file=args.model_file)
    model.eval()
    return model_cfg, model, dataset, device


def vis_mesh_prediction(pred_mesh: trimesh.Trimesh, gt_mesh: trimesh.Trimesh, pred_surface_contact: np.ndarray,
                        gt_surface_contact: np.ndarray):
    plt = Plotter(shape=(1, 3))

    _, gt_contact_triangles, _ = mesh_utils.find_in_contact_triangles(gt_mesh, gt_surface_contact)
    _, pred_contact_triangles, _ = mesh_utils.find_in_contact_triangles(gt_mesh, pred_surface_contact)

    gt_mesh_vedo = Mesh([gt_mesh.vertices, gt_mesh.faces])
    gt_colors = [[255, 0, 0, 255] if c else [255, 255, 0, 255] for c in gt_contact_triangles]
    gt_mesh_vedo.celldata["CellIndividualColors"] = np.array(gt_colors).astype(np.uint8)
    gt_mesh_vedo.celldata.select("CellIndividualColors")
    plt.at(0).show(gt_mesh_vedo, vedo_utils.draw_origin(), "GT Mesh")

    pred_surface_mesh = Mesh([gt_mesh.vertices, gt_mesh.faces])
    pred_colors = [[255, 0, 0, 255] if c else [255, 255, 0, 255] for c in pred_contact_triangles]
    pred_surface_mesh.celldata["CellIndividualColors"] = np.array(pred_colors).astype(np.uint8)
    pred_surface_mesh.celldata.select("CellIndividualColors")
    plt.at(1).show(pred_surface_mesh, vedo_utils.draw_origin(), "Pred. Contact")

    pred_geom_mesh = Mesh([pred_mesh.vertices, pred_mesh.faces])
    plt.at(2).show(pred_geom_mesh, vedo_utils.draw_origin(), "Pred. Mesh")

    plt.interactive().close()


def test_inference_perf(args):
    dataset: ToolDataset
    model_cfg, model, dataset, device = load_model_dataset_from_args(args)

    vis = args.vis

    # Load meshes.
    gt_meshes = []
    dataset_dir = dataset.dataset_dir
    for trial_idx in range(len(dataset)):
        mesh_fn = os.path.join(dataset_dir, "out_%d_mesh.obj" % trial_idx)
        mesh = trimesh.load(mesh_fn)
        gt_meshes.append(mesh)

    out_dir = args.out
    if out_dir is not None:
        mmint_utils.make_dir(out_dir)

    metrics_results = []

    for trial_idx in range(len(dataset)):
        trial_dict = dataset[trial_idx]
        latent_code, pred_dict, surface_pred_dict, mesh = infer_latent_from_surface(model, trial_dict, {},
                                                                                    device=device)

        # Compare meshes.
        if vis:
            vis_mesh_prediction(mesh, gt_meshes[trial_idx],
                                trial_dict["surface_points"][surface_pred_dict["in_contact"].cpu().squeeze() > 0.5],
                                trial_dict["surface_points"][trial_dict["surface_in_contact"]])

            results_dict = {
                "gt": numpy_dict(trial_dict),
                "pred": numpy_dict(pred_dict)
            }
            vis_prediction_vs_dataset(results_dict, mesh=mesh)

        # Write results.
        if out_dir is not None:
            # Write predicted mesh to file.
            mesh.export(os.path.join(out_dir, "pred_%d_mesh.obj" % trial_idx))

            # Write surface predictions to file.
            mmint_utils.save_gzip_pickle(surface_pred_dict,
                                         os.path.join(out_dir, "pred_%d_surface.pkl.gzip" % trial_idx))

        # Calculate metrics.
        binary_accuracy = ncf_metrics.binary_accuracy(surface_pred_dict["in_contact"].cpu().squeeze() > 0.5,
                                                      torch.from_numpy(trial_dict["surface_in_contact"]))
        pr = ncf_metrics.precision_recall(surface_pred_dict["in_contact"].cpu().squeeze() > 0.5,
                                          torch.from_numpy(trial_dict["surface_in_contact"]))
        chamfer_dist = ncf_metrics.mesh_chamfer_distance(mesh, gt_meshes[trial_idx])

        metrics_results.append({
            "binary_accuracy": binary_accuracy,
            "pr": pr,
            "chamfer_distance": chamfer_dist,
        })

    # Write all metrics to file.
    if out_dir is not None:
        mmint_utils.save_gzip_pickle(metrics_results, os.path.join(out_dir, "metrics.pkl.gzip"))


if __name__ == '__main__':
    parser_ = get_model_dataset_arg_parser()
    args_ = parser_.parse_args()
    test_inference_perf(args_)
