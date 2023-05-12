import argparse
import os
import random

from vedo import Plotter, Points, LegendBox

import mmint_utils
import numpy as np
import torch
import neural_contact_fields.metrics as ncf_metrics
from neural_contact_fields import config
from neural_contact_fields.utils import utils, vedo_utils
from neural_contact_fields.utils.results_utils import load_gt_results, load_pred_results, print_results
from tqdm import trange, tqdm
import torchmetrics
import pytorch3d.loss


def eval_example(gen_dict, gt_dict, device, sample: bool = True, verbose: bool = False):
    metrics_dict = dict()

    # Evaluate meshes.
    if gen_dict["mesh"] is not None and "mesh" in gt_dict:
        chamfer_dist = ncf_metrics.mesh_chamfer_distance(gen_dict["mesh"], gt_dict["mesh"], device=device, vis=False)
        iou = ncf_metrics.mesh_iou(gt_dict["points_iou"], gt_dict["iou_labels"], gen_dict["mesh"], device=device,
                                   vis=False)
        metrics_dict.update({
            "chamfer_distance": chamfer_dist.item() * 1e6,
            "iou": iou.item(),
        })

    # Evaluate pointclouds.
    if gen_dict["pointcloud"] is not None and "pointcloud" in gt_dict:
        if sample:
            pred_pointcloud = utils.sample_pointcloud(gen_dict["pointcloud"], 10000)
        else:
            pred_pointcloud = gen_dict["pointcloud"]
        gt_pc = utils.sample_pointcloud(gt_dict["pointcloud"], 10000)
        chamfer_dist, _ = pytorch3d.loss.chamfer_distance(pred_pointcloud.unsqueeze(0).float(),
                                                          gt_pc.unsqueeze(0).float())

        metrics_dict.update({
            "chamfer_distance": chamfer_dist.item() * 1e6,
        })

    # Evaluate contact patches.
    if gen_dict["contact_patch"] is not None:
        # Sample each to 300 - makes evaluation of CD more fair.
        if sample:
            pred_pc = utils.sample_pointcloud(gen_dict["contact_patch"], 300)
        else:
            pred_pc = gen_dict["contact_patch"]
        if len(pred_pc) > 0:
            if type(pred_pc) == np.ndarray:
                pred_pc = torch.from_numpy(pred_pc).to(device)
            gt_pc = utils.sample_pointcloud(gt_dict["contact_patch"], 300)

            if verbose:
                plt = Plotter()
                pred_patch_pc = Points(pred_pc.cpu().numpy(), c="red").legend("Predicted")
                gt_patch_pc = Points(gt_pc.cpu().numpy(), c="blue").legend("Ground Truth")
                leg = LegendBox([pred_patch_pc, gt_patch_pc])
                plt.at(0).show(pred_patch_pc, gt_patch_pc, leg, vedo_utils.draw_origin(), "Pred. Contact Patch")

            patch_chamfer_dist, _ = pytorch3d.loss.chamfer_distance(pred_pc.unsqueeze(0).float(),
                                                                    gt_pc.unsqueeze(0).float())

            metrics_dict.update({
                "patch_chamfer_distance": patch_chamfer_dist.item() * 1e6,
            })
        else:
            metrics_dict.update({
                "patch_chamfer_distance": None,
            })

    # Evaluate binary contact labels.
    if gen_dict["contact_labels"] is not None:
        pred_contact_labels_trial = gen_dict["contact_labels"]["contact_labels"].float()
        gt_contact_label = gt_dict["contact_labels"].float()
        binary_accuracy = torchmetrics.functional.classification.binary_accuracy(pred_contact_labels_trial,
                                                                                 gt_contact_label,
                                                                                 threshold=0.5)
        precision = torchmetrics.functional.classification.binary_precision(pred_contact_labels_trial,
                                                                            gt_contact_label,
                                                                            threshold=0.5)
        recall = torchmetrics.functional.classification.binary_recall(pred_contact_labels_trial,
                                                                      gt_contact_label,
                                                                      threshold=0.5)
        f1 = torchmetrics.functional.classification.binary_f1_score(pred_contact_labels_trial,
                                                                    gt_contact_label,
                                                                    threshold=0.5)
        metrics_dict.update({
            "binary_accuracy": binary_accuracy.item(),
            "precision": precision.item(),
            "recall": recall.item(),
            "f1": f1.item(),
        })

    if "iou_labels" in gen_dict:  # TODO: Fix key issue.
        if gen_dict["iou_labels"] is not None:
            pred_iou_labels_trial = gen_dict["iou_labels"]["iou_labels"].float()
            gt_iou_labels_trial = gt_dict["iou_labels"].float()

            iou = torchmetrics.functional.classification.binary_jaccard_index(pred_iou_labels_trial,
                                                                              gt_iou_labels_trial,
                                                                              threshold=0.5)

            metrics_dict.update({
                "model_iou": iou.item(),
            })

    if gen_dict["metadata"] is not None:
        for key in ["mesh_gen_time", "latent_gen_time", "iters"]:
            if key in gen_dict["metadata"]:
                metrics_dict[key] = gen_dict["metadata"][key]

    res_dict = {
        "metadata": {
            "env_id": gt_dict["env_class"]  # Add environment class to metrics dict - allows us to filter later.
        },
        "metrics": metrics_dict,
    }
    return res_dict


def calculate_metrics(dataset_cfg_fn: str, dataset_mode: str, out_dir: str, verbose: bool = False,
                      sample: bool = False):
    device = torch.device("cuda:0")

    # Load dataset.
    dataset_config = mmint_utils.load_cfg(dataset_cfg_fn)
    dataset = config.get_dataset(dataset_mode, dataset_config)
    num_trials = len(dataset)

    # Check if there are multiple runs.
    run_dirs = [f for f in os.listdir(out_dir) if "run_" in f]
    if len(run_dirs) == 0:
        run_dirs = ["./"]

    with tqdm(total=len(run_dirs) * num_trials) as pbar:
        for run_dir in run_dirs:
            run_out_dir = os.path.join(out_dir, run_dir)

            # Load specific ground truth results needed for evaluation.
            gt_dicts = load_gt_results(dataset, num_trials, device)

            # Load predicted results.
            gen_dicts = load_pred_results(run_out_dir, num_trials, device)

            # Calculate metrics.
            metrics_results = []
            for trial_idx in range(num_trials):
                metrics_dict = eval_example(gen_dicts[trial_idx], gt_dicts[trial_idx], device, sample=sample,
                                            verbose=False)
                metrics_results.append(metrics_dict)
                pbar.update()

            if verbose:
                print_results(metrics_results, os.path.dirname(run_out_dir))

            # Write all metrics to file.
            if run_out_dir is not None:
                mmint_utils.save_gzip_pickle(metrics_results, os.path.join(run_out_dir, "metrics.pkl.gzip"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate metrics on generated data.")
    parser.add_argument("dataset_cfg", type=str, help="Dataset configuration.")
    parser.add_argument("out_dir", type=str, help="Out directory where results are written to.")
    parser.add_argument("--mode", "-m", type=str, default="test", help="Dataset mode [train, val, test]")
    parser.add_argument("--verbose", "-v", dest='verbose', action='store_true', help='Be verbose.')
    parser.set_defaults(verbose=False)
    parser.add_argument("--sample", "-s", dest='sample', action='store_true',
                        help='Sample pointclouds to set number before evaluation.')
    parser.set_defaults(sample=False)
    args = parser.parse_args()

    # Seed for repeatability.
    torch.manual_seed(10)
    np.random.seed(10)
    random.seed(10)

    calculate_metrics(args.dataset_cfg, args.mode, args.out_dir, args.verbose, args.sample)
