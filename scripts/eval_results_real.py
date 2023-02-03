import argparse
import os
import random

import mmint_utils
import numpy as np
import torch
import neural_contact_fields.metrics as ncf_metrics
from neural_contact_fields import config
from neural_contact_fields.utils.results_utils import load_gt_results_real, print_results, load_pred_results
from tqdm import trange
import torchmetrics
import pytorch3d.loss


def calculate_metrics(dataset_cfg_fn: str, dataset_mode: str, out_dir: str):
    device = torch.device("cuda:0")

    raise Exception("Setup sampling properly!")

    # Load dataset.
    dataset_config = mmint_utils.load_cfg(dataset_cfg_fn)
    dataset = config.get_dataset(dataset_mode, dataset_config)
    num_trials = len(dataset)
    dataset_dir = dataset_config["data"][dataset_mode]["dataset_dir"]

    # Load specific ground truth results needed for evaluation.
    gt_contact_patches = load_gt_results_real(dataset, dataset_dir, num_trials, device)

    # Load predicted results.
    pred_meshes, pred_pointclouds, pred_contact_patches, _ = load_pred_results(out_dir, num_trials, device)

    # Calculate metrics.
    metrics_results = []
    for trial_idx in trange(num_trials):
        metrics_dict = dict()

        # Evaluate contact patches.
        if pred_contact_patches[trial_idx] is not None:
            patch_chamfer_dist, _ = pytorch3d.loss.chamfer_distance(
                pred_contact_patches[trial_idx].unsqueeze(0).float(),
                gt_contact_patches[trial_idx].unsqueeze(0).float())

            metrics_dict.update({
                "patch_chamfer_distance": patch_chamfer_dist.item(),
            })

        metrics_results.append(metrics_dict)

    # Write all metrics to file.
    if out_dir is not None:
        mmint_utils.save_gzip_pickle(metrics_results, os.path.join(out_dir, "metrics.pkl.gzip"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate metrics on generated data.")
    parser.add_argument("dataset_cfg", type=str, help="Dataset configuration.")
    parser.add_argument("out_dir", type=str, help="Out directory where results are written to.")
    parser.add_argument("--mode", "-m", type=str, default="test", help="Dataset mode [train, val, test]")
    args = parser.parse_args()

    # Seed for repeatability.
    torch.manual_seed(10)
    np.random.seed(10)
    random.seed(10)

    calculate_metrics(args.dataset_cfg, args.mode, args.out_dir)
