import argparse
import os

import mmint_utils
import torch
import trimesh
from neural_contact_fields.explicit_baseline.grnet.extensions.chamfer_dist import ChamferDistance
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
    _, pred_pcds, pred_contact_patches, _ = load_pred_results(out_dir, num_trials)

    # Calculate metrics.
    metrics_results = []
    for trial_idx in trange(num_trials):

        pcd_chamfer_distance = ChamferDistance()
        chamfer_dist = pcd_chamfer_distance(pred_meshes[trial_idx], gt_meshes[trial_idx], device=device)
        
        metrics_results.append({
            "chamfer_distance": chamfer_dist,
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
