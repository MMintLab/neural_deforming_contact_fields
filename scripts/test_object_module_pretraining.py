import argparse
import os.path
import pdb

import torch.nn as nn
import mmint_utils
import torch
from matplotlib import pyplot as plt
from neural_contact_fields.utils.model_utils import load_model_and_dataset, load_model
import numpy as np
from scripts.vis_object_module_pretraining import vis_object_module_pretraining
from tqdm import trange


def get_model_dataset_arg_parser():
    """
    Argument parser for common model + dataset arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Model/data config file.")
    parser.add_argument("out_fn", type=str, help="Place to write results to.")
    parser.add_argument("-v", "--vis", dest="vis", action="store_true", help="Visualize.")
    parser.set_defaults(vis=False)
    return parser


def load_model_dataset_from_args(args):
    """
    Load model and dataset from arguments object.
    """
    model_cfg, model, dataset, device = load_model_and_dataset(args.config, dataset_mode="pretrain",
                                                               model_file="pretrain_model.pt")
    model.eval()

    num_objects = len(dataset)
    object_code = nn.Embedding(num_objects, model_cfg["model"]["z_object_size"]).to(device)
    model_file = os.path.join(model_cfg['pretraining']['out_dir'], "pretrain_model.pt")
    load_model({"object_code": object_code}, model_file)

    return model_cfg, model, object_code, dataset, device


def test_object_module_inference(args):
    vis = args.vis
    max_batch = 40 ** 3
    model_cfg, model, object_code, dataset, device = load_model_dataset_from_args(args)
    model.eval()

    # Predict on query points.
    z_object = object_code(torch.tensor(0, device=device)).to(device).float()
    head = 0

    batch = dataset[0]
    query_points = torch.from_numpy(batch["query_point"]).float().to(device)
    num_samples = query_points.shape[0]

    num_iters = int(np.ceil(num_samples / max_batch))
    sdf_preds = []
    normals_preds = []

    for iter_idx in trange(num_iters):
        sample_subset = query_points[head: min(head + max_batch, num_samples), 0:3]

        out_dict = model.forward_object_module(sample_subset.float().unsqueeze(0), z_object.unsqueeze(0))
        sdf_preds.append(out_dict["sdf"].squeeze(0))
        normals_preds.append(out_dict["normals"].squeeze(0))

        head += max_batch
    sdf_preds = torch.cat(sdf_preds, dim=0)
    normals_preds = torch.cat(normals_preds, dim=0)

    # Write prediction results to file.
    out_fn = args.out_fn
    mmint_utils.make_dir(os.path.dirname(out_fn))

    out_dict = {
        "query_points": batch["query_point"],
        "pred_sdf": sdf_preds.detach().cpu().numpy(),
        "pred_normals": normals_preds.detach().cpu().numpy(),
        "sdf": batch["sdf"],
    }

    mmint_utils.save_gzip_pickle(out_dict, out_fn)

    if vis:
        vis_object_module_pretraining(out_dict)

    # Visualize a slice through the object.
    y = 0
    x_min = -0.04
    x_max = 0.04
    z_min = 0.02
    z_max = 0.1
    xs = np.arange(x_min, x_max, (x_max - x_min) / 1000.0)
    zs = np.arange(z_min, z_max, (z_max - z_min) / 1000.0)

    # Build query points.
    query_points = []
    for x in xs:
        for z in zs:
            query_points.append([x, y, z])
    qp = np.array(query_points)
    qp = torch.from_numpy(qp).to(device).unsqueeze(0).float()

    slice_out_dict = model.forward_object_module(qp, z_object=z_object)
    slice_pred_sdf = slice_out_dict["sdf"].detach().cpu().numpy()

    slice_image = np.zeros([len(zs), len(xs)])
    for idx in range(len(query_points)):
        z = int(idx // len(zs))
        x = int(idx % len(zs))

        slice_image[z, x] = slice_pred_sdf[0, idx]

    plt.imshow(slice_image)
    plt.show()


if __name__ == '__main__':
    parser = get_model_dataset_arg_parser()
    args_ = parser.parse_args()

    test_object_module_inference(args_)
