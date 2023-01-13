import argparse
import os

import mmint_utils
import torch
import trimesh
import neural_contact_fields.metrics as ncf_metrics
from tqdm import trange


def calculate_metrics(dataset_dir: str, out_dir: str):
    device = torch.device("cuda:0")

    # Load ground truth meshes and surface contact labels.
    data_fns = sorted([f for f in os.listdir(dataset_dir) if "out" in f and ".pkl.gzip" in f],
                      key=lambda x: int(x.split(".")[0].split("_")[-1]))
    num_trials = len(data_fns)
    gt_meshes = []
    gt_surface_contact_labels = []
    for trial_idx in range(num_trials):
        data_dict = mmint_utils.load_gzip_pickle(os.path.join(dataset_dir, data_fns[trial_idx]))
        gt_surface_contact_labels.append(data_dict["test"]["surface_in_contact"])

        gt_meshes.append(trimesh.load(os.path.join(dataset_dir, "out_%d_mesh.obj" % trial_idx)))

    # Load generated data.
    pred_meshes = []
    pred_surface_contact_labels = []
    for trial_idx in range(num_trials):
        data_dict = mmint_utils.load_gzip_pickle(os.path.join(out_dir, "pred_%d_surface.pkl.gzip" % trial_idx))
        pred_surface_contact_labels.append(data_dict["in_contact"].detach().cpu().squeeze().numpy())

        pred_meshes.append(trimesh.load(os.path.join(out_dir, "pred_%d_mesh.obj" % trial_idx)))

    # Calculate metrics.
    metrics_results = []
    for trial_idx in trange(num_trials):
        binary_accuracy = ncf_metrics.binary_accuracy(torch.from_numpy(pred_surface_contact_labels[trial_idx] > 0.5),
                                                      torch.from_numpy(gt_surface_contact_labels[trial_idx]))
        pr = ncf_metrics.precision_recall(torch.from_numpy(pred_surface_contact_labels[trial_idx] > 0.5),
                                          torch.from_numpy(gt_surface_contact_labels[trial_idx]))
        chamfer_dist = ncf_metrics.mesh_chamfer_distance(pred_meshes[trial_idx], gt_meshes[trial_idx])

        metrics_results.append({
            "binary_accuracy": binary_accuracy,
            "pr": pr,
            "chamfer_distance": chamfer_dist,
        })

    # Write all metrics to file.
    if out_dir is not None:
        mmint_utils.save_gzip_pickle(metrics_results, os.path.join(out_dir, "metrics.pkl.gzip"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate metrics on generated data.")
    parser.add_argument("dataset_dir", type=str, help="Dataset directory.")
    parser.add_argument("out_dir", type=str, help="Out directory where results are written to.")
    args = parser.parse_args()

    calculate_metrics(args.dataset_dir, args.out_dir)
