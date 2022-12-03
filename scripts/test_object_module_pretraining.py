import argparse
import os.path

import mmint_utils
import torch
from neural_contact_fields.utils.utils import load_model_and_dataset
import numpy as np
from tqdm import trange


def get_model_dataset_arg_parser():
    """
    Argument parser for common model + dataset arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Model/data config file.")
    parser.add_argument("out_fn", type=str, help="Place to write results to.")
    return parser


def load_model_dataset_from_args(args):
    """
    Load model and dataset from arguments object.
    """
    model_cfg, model, dataset, device = load_model_and_dataset(args.config, dataset_mode="test", model_file="model.pt")
    model.eval()

    return model_cfg, model, dataset, device


def test_object_module_inference(args):
    max_batch = 40 ** 3
    model_cfg, model, dataset, device = load_model_dataset_from_args(args)

    # Predict on query points.
    head = 0
    num_samples = len(dataset)
    num_iters = int(np.ceil(num_samples / max_batch))
    sdf_preds = []
    for iter_idx in trange(num_iters):
        sample_subset = torch.from_numpy(dataset.query_points[head: min(head + max_batch, num_samples), 0:3]).to(device)

        with torch.no_grad():
            sdf_pred = model.forward_object_module(sample_subset.float())
        sdf_preds.append(sdf_pred)

        head += max_batch
    sdf_preds = torch.cat(sdf_preds, dim=0)

    # Write prediction results to file.
    out_fn = args.out_fn
    mmint_utils.make_dir(os.path.dirname(out_fn))

    out_dict = {
        "query_points": dataset.query_points,
        "pred_sdf": sdf_preds.cpu().numpy(),
        "sdf": dataset.sdf,
    }

    mmint_utils.save_gzip_pickle(out_dict, out_fn)


if __name__ == '__main__':
    parser = get_model_dataset_arg_parser()
    args_ = parser.parse_args()

    test_object_module_inference(args_)
