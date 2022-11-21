import argparse

import mmint_utils
import numpy as np
import torch
from neural_contact_fields import vedo_utils
from neural_contact_fields.data.tool_dataset import ToolDataset
from neural_contact_fields.inference import points_inference_latent
from neural_contact_fields.model_utils import load_model_and_dataset

from vedo import Plotter, Points, Arrows


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


def test_interpolation(args):
    global points
    global defs
    global contact_points

    dataset: ToolDataset
    model_cfg, model, dataset, device = load_model_dataset_from_args(args)

    # Get some query points:
    trial_dict = dataset.get_all_points_for_trial(None, 0)
    query_points = torch.from_numpy(trial_dict["query_point"]).to(device).float()
    query_points_np = query_points.cpu().numpy()

    # Get latent codes to interpolate.
    latent_0 = model.trial_to_latent(torch.zeros([1], dtype=torch.int32, device=device))
    latent_1 = model.trial_to_latent(torch.ones([1], dtype=torch.int32, device=device))

    res_dict = points_inference_latent(model, latent_0, query_points, device=device)
    pred_sdf = res_dict["sdf"].cpu().numpy()
    pred_def = res_dict["delta_coords"].cpu().numpy()
    pred_contact = res_dict["contact_prob"].cpu().numpy() > 0.5

    # Create plotter.
    plt = Plotter(shape=(1, 3))
    points = Points(query_points_np[pred_sdf <= 0.0], c="b")
    plt.at(0).show(points,
                   vedo_utils.draw_origin(), "Predicted Surface")
    defs = Arrows(query_points_np, query_points_np - pred_def)
    plt.at(1).show(defs, "Predicted Deformations")
    contact_points = Points(query_points_np[np.logical_and(pred_sdf <= 0.0, pred_contact)], c="r")
    plt.at(2).show(points,
                   contact_points,
                   vedo_utils.draw_origin(), "Predicted Contact")

    def update_ncf(widget, event):
        global points
        global defs
        global contact_points

        interpolate_val = widget.value
        new_latent = latent_0 + ((latent_1 - latent_0) * interpolate_val)

        res_dict = points_inference_latent(model, new_latent, query_points, device=device)
        pred_sdf = res_dict["sdf"].cpu().numpy()
        pred_def = res_dict["delta_coords"].cpu().numpy()
        pred_contact = res_dict["contact_prob"].cpu().numpy() > 0.5

        plt.at(0).clear(points)
        plt.at(2).clear([points, contact_points])
        plt.at(1).clear(defs)

        points = Points(query_points_np[pred_sdf <= 0.0], c="b")
        plt.at(0).add(points)

        defs = Arrows(query_points_np, query_points_np - pred_def)
        plt.at(1).add(defs)

        contact_points = Points(query_points_np[np.logical_and(pred_sdf <= 0.0, pred_contact)], c="r")
        plt.at(2).add([points, contact_points])

    plt.add_slider(update_ncf, xmin=0.0, xmax=1.0, value=0.5, title="Interpolation",
                   alpha=1.0,
                   pos=[[0.1, 0.1], [0.2, 0.1]])
    plt.show().interactive().close()


if __name__ == '__main__':
    parser = get_model_dataset_arg_parser()
    args_ = parser.parse_args()

    test_interpolation(args_)
