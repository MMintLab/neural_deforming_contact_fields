import os.path
import pdb

import mmint_utils
import numpy as np
import torch
from neural_contact_fields.data.tool_dataset import ToolDataset
from neural_contact_fields.inference import points_inference
from neural_contact_fields.model_utils import load_model_and_dataset
import neural_contact_fields.vis as vis
import argparse


def get_model_dataset_arg_parser():
    """
    Argument parser for common model + dataset arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Model/data config file.")
    parser.add_argument("out", type=str, help="Directory to write results to.")
    parser.add_argument("--dataset_config", "-d", type=str, default=None, help="Optional dataset config to use.")
    parser.add_argument("--mode", "-m", type=str, default="test", help="Which split to vis [train, val, test].")
    parser.add_argument("--model_file", "-f", type=str, default="model.pt", help="Which model save file to use.")
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


def numpy_dict(torch_dict: dict):
    np_dict = dict()
    for k, v in torch_dict.items():
        if type(v) is torch.Tensor:
            np_dict[k] = v.cpu().numpy()
        else:
            np_dict[k] = v
    return np_dict


def test_inference(args):
    dataset: ToolDataset
    model_cfg, model, dataset, device = load_model_dataset_from_args(args)

    out_dir = args.out
    mmint_utils.make_dir(out_dir)

    trial_indices = dataset.get_trial_indices()
    for trial_idx in trial_indices:
        trial_dict = dataset.get_all_points_for_trial(None, trial_idx)
        trial_pred_dict = points_inference(model, trial_dict, device=device)

        out_fn = os.path.join(out_dir, "pred_%d.pkl.gzip" % trial_idx)
        mmint_utils.save_gzip_pickle({
            "gt": numpy_dict(trial_dict),
            "pred": numpy_dict(trial_pred_dict)
        }, out_fn)


if __name__ == '__main__':
    parser = get_model_dataset_arg_parser()
    args_ = parser.parse_args()

    test_inference(args_)
