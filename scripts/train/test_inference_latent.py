import os

import mmint_utils
import numpy as np
import torch
from neural_contact_fields.data.tool_dataset import ToolDataset
from neural_contact_fields.inference import points_inference
from neural_contact_fields.utils.model_utils import load_model_and_dataset, load_model
import argparse

from neural_contact_fields.utils.utils import numpy_dict
from vis_prediction_vs_dataset import vis_prediction_vs_dataset
import torch.nn as nn


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
    dataset: ToolDataset
    model_cfg, model, dataset, device = load_model_and_dataset(args.config, dataset_config=args.dataset_config,
                                                               dataset_mode=args.mode,
                                                               model_file=args.model_file)
    model.eval()
    return model_cfg, model, dataset, device


def test_inference(args):
    vis = args.vis
    dataset: ToolDataset
    model_cfg, model, dataset, device = load_model_dataset_from_args(args)

    out_dir = args.out
    if out_dir is not None:
        mmint_utils.make_dir(out_dir)

    num_trials = dataset.get_num_trials()
    for trial_idx in range(num_trials):
        trial_dict = dataset[trial_idx]
        trial_pred_dict = points_inference(model, trial_dict, device=device)
        results_dict = {
            "gt": numpy_dict(trial_dict),
            "pred": numpy_dict(trial_pred_dict)
        }

        if vis:
            vis_prediction_vs_dataset(results_dict)

        if out_dir is not None:
            out_fn = os.path.join(out_dir, "pred_%d.pkl.gzip" % trial_idx)
            mmint_utils.save_gzip_pickle(results_dict, out_fn)


if __name__ == '__main__':
    parser = get_model_dataset_arg_parser()
    args_ = parser.parse_args()

    test_inference(args_)
