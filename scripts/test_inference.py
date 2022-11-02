import mmint_utils
import numpy as np
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
    parser.add_argument("--dataset_config", "-d", type=str, default=None, help="Optional dataset config to use.")
    parser.add_argument("--mode", "-m", type=str, default="test", help="Which split to vis [train, val, test].")
    parser.add_argument("--model_file", "-f", type=str, default="model_best.pt", help="Which model save file to use.")
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


def test_inference(args):
    model_cfg, model, dataset, device = load_model_dataset_from_args(args)

    samples = points_inference(model, device=device)

    print("Inference done.")

    # points = samples[samples[:, 3] <= 0.03]
    # colors = np.zeros([points.shape[0], 4], dtype=np.float32)
    # colors[:, 3] = 1.0
    # colors[:, 2] = 1.0
    # colors[points[:, 4] >= 0.5, :] = 0.0
    # colors[points[:, 4] >= 0.5, 0] = 1.0
    # colors[points[:, 4] >= 0.5, 3] = 1.0
    # vis.plot_points(points, colors=colors)

    # mmint_utils.save_gzip_pickle(samples, "test_inference.pkl.gzip")


if __name__ == '__main__':
    parser = get_model_dataset_arg_parser()
    args_ = parser.parse_args()

    test_inference(args_)
