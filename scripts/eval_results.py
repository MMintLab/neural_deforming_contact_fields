import argparse
import os

import mmint_utils
import torch
import trimesh
import neural_contact_fields.metrics as ncf_metrics
from neural_contact_fields import config
from neural_contact_fields.utils.results_utils import load_gt_results, load_pred_results, print_results
from tqdm import trange


def calculate_metrics(dataset_cfg_fn: str, dataset_mode: str, out_dir: str, verbose: bool = False):
    device = torch.device("cuda:0")

    # Load dataset.
    dataset_config = mmint_utils.load_cfg(dataset_cfg_fn)
    dataset = config.get_dataset(dataset_mode, dataset_config)
    num_trials = len(dataset)
    dataset_dir = dataset_config["data"][dataset_mode]["dataset_dir"]

    # Load specific ground truth results needed for evaluation.
    gt_meshes, _, _, gt_contact_labels, points_iou, gt_occ_iou = load_gt_results(dataset, dataset_dir, num_trials)

    # Load predicted results.
    pred_meshes, _, _, pred_contact_labels = load_pred_results(out_dir, num_trials)

    # Calculate metrics.
    metrics_results = []
    for trial_idx in trange(num_trials):
        binary_accuracy = ncf_metrics.binary_accuracy(pred_contact_labels[trial_idx] > 0.5,
                                                      torch.from_numpy(gt_contact_labels[trial_idx]).to(device))
        pr = ncf_metrics.precision_recall(pred_contact_labels[trial_idx] > 0.5,
                                          torch.from_numpy(gt_contact_labels[trial_idx]).to(device))
        chamfer_dist = ncf_metrics.mesh_chamfer_distance(pred_meshes[trial_idx], gt_meshes[trial_idx], device=device)
        iou = ncf_metrics.mesh_iou(torch.from_numpy(points_iou[trial_idx]).to(device),
                                   torch.from_numpy(gt_occ_iou[trial_idx]).to(device),
                                   pred_meshes[trial_idx], device=device)

        metrics_results.append({
            "binary_accuracy": binary_accuracy,
            "pr": pr,
            "chamfer_distance": chamfer_dist,
            "iou": iou.item(),
        })

    if verbose:
        print_results(metrics_results, os.path.dirname(out_dir))

    # Write all metrics to file.
    if out_dir is not None:
        mmint_utils.save_gzip_pickle(metrics_results, os.path.join(out_dir, "metrics.pkl.gzip"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate metrics on generated data.")
    parser.add_argument("dataset_cfg", type=str, help="Dataset configuration.")
    parser.add_argument("out_dir", type=str, help="Out directory where results are written to.")
    parser.add_argument("--mode", "-m", type=str, default="test", help="Dataset mode [train, val, test]")
    parser.add_argument("--verbose", "-v", dest='verbose', action='store_true', help='Be verbose.')
    parser.set_defaults(verbose=False)
    args = parser.parse_args()

    calculate_metrics(args.dataset_cfg, args.mode, args.out_dir, args.verbose)
