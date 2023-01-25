import argparse
import os

import mmint_utils
import numpy as np
import torch
import torchmetrics.functional.classification
import tqdm
from matplotlib import pyplot as plt
from neural_contact_fields import config
from neural_contact_fields.utils.model_utils import load_model_and_dataset
from neural_contact_fields.utils.results_utils import load_pred_results, write_results
from tqdm import trange


def tune_contact_threshold(model_cfg, model, val_dataset, out_dir: str = None, device=None, load_dict: dict = None,
                           vis: bool = False):
    model.eval()
    dataset_size = len(val_dataset)

    # Load generator.
    generator = config.get_generator(model_cfg, model, {}, device)

    if not generator.generates_contact_labels:
        print("Selected model does not generate contact labels - nothing to do!")
        return

    print("Loading/Generating contact labels...")

    # If out directory provided, check if data already exists.
    if out_dir is not None and os.path.exists(out_dir):
        _, _, _, pred_contact_dicts_all = load_pred_results(out_dir, dataset_size, device)
        pred_contact_labels_all = [pred_dict["contact_labels"] for pred_dict in pred_contact_dicts_all]
    elif out_dir is not None and not os.path.exists(out_dir):
        mmint_utils.make_dir(out_dir)
        pred_contact_labels_all = [None] * dataset_size
    else:
        pred_contact_labels_all = [None] * dataset_size

    # Go through validation dataset and generate contact labels.
    gt_contact_labels_all = []
    for idx in trange(dataset_size):
        data_dict = val_dataset[idx]

        # Get GT contact labels from dataset.
        gt_contact_labels_all.append(torch.from_numpy(data_dict["surface_in_contact"]).int().to(device))

        # If data has already been generated, skip generation step.
        if pred_contact_labels_all[idx] is not None:
            continue

        # Generate contact labels for example.
        pred_contact_labels, _ = generator.generate_contact_labels(data_dict, {})
        pred_contact_labels_all[idx] = pred_contact_labels["contact_labels"]

        # Save generated contact labels to file.
        if out_dir is not None:
            write_results(out_dir, None, None, None, pred_contact_labels, idx)

    pred_contact_labels_all = torch.cat(pred_contact_labels_all, dim=0)
    gt_contact_labels_all = torch.cat(gt_contact_labels_all, dim=0)

    # Search through contact thresholds for best performance.
    print("Searching thresholds...")
    thresholds = np.arange(0.0, 1.000000001, 0.005)

    # F1 Score.
    f1_scores = []
    for threshold in tqdm.tqdm(thresholds):
        f1 = torchmetrics.functional.classification.binary_f1_score(pred_contact_labels_all, gt_contact_labels_all,
                                                                    threshold, multidim_average="global")
        f1_scores.append(f1.item())
    if vis:
        plt.plot(thresholds, f1_scores)
        plt.xlim(0.0, 1.0)
        plt.xlabel("Binary Threshold")
        plt.ylim(0.0, 1.0)
        plt.ylabel("F1 Score")
        plt.title("F1 Scores")
        plt.show()

    # PR Curve.
    if vis:
        precisions, recalls, _ = torchmetrics.functional.classification.binary_precision_recall_curve(
            pred_contact_labels_all, gt_contact_labels_all, thresholds=list(thresholds))
        plt.plot(recalls.cpu().numpy(), precisions.cpu().numpy())
        plt.scatter(recalls.cpu().numpy(), precisions.cpu().numpy())
        plt.xlim(0.0, 1.0)
        plt.xlabel("Recall")
        plt.ylim(0.0, 1.0)
        plt.ylabel("Precision")
        plt.title("PR Curve")
        plt.show()

    # Determine best threshold using f1 scores.
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1_score = f1_scores[best_idx]
    print("Best threshold: %f. F1 Score: %f." % (best_threshold, best_f1_score))

    # Save threshold to model file. This allows it to be easily loaded at test time.
    model_dict = {
        "model": model,
        "generation": {
            "contact_threshold": best_threshold,
        },
    }
    model_dict.update(load_dict)
    model_dir = model_cfg["training"]["out_dir"]
    torch.save(model_dict, os.path.join(model_dir, "model_threshold.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Model/data config file.")
    parser.add_argument("val_dataset_config", type=str, help="Validation dataset config to use.")
    parser.add_argument("--model_file", "-f", type=str, default="model.pt", help="Which model save file to use.")
    parser.add_argument("--out_dir", "-o", type=str, default=None,
                        help="Optional out dir to load/save generated labels used.")
    parser.add_argument("--vis", "-v", action='store_true', help="Visualize results.")
    parser.set_defaults(vis=False)
    args = parser.parse_args()

    model_cfg_, model_, val_dataset_, device_, load_dict_ = load_model_and_dataset(
        args.config, dataset_config=args.val_dataset_config, dataset_mode="val", model_file=args.model_file
    )

    tune_contact_threshold(model_cfg_, model_, val_dataset_, args.out_dir, device_, load_dict_, args.vis)
