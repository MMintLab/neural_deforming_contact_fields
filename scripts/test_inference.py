import mmint_utils
import torch
from neural_contact_fields.data.tool_dataset import ToolDataset
from neural_contact_fields.inference import infer_latent, points_inference_latent
from neural_contact_fields.utils.utils import load_model_and_dataset
import argparse
from vis_prediction_vs_dataset import vis_prediction_vs_dataset


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
            np_dict[k] = v.detach().cpu().numpy()
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
        query_points = torch.from_numpy(trial_dict["query_point"]).to(device).float()
        latent_code = infer_latent(model, trial_dict, device=device)

        res_dict = points_inference_latent(model, latent_code, query_points, device=device)

        results_dict = {
            "gt": numpy_dict(trial_dict),
            "pred": numpy_dict(res_dict)
        }

        vis_prediction_vs_dataset(results_dict)


if __name__ == '__main__':
    parser = get_model_dataset_arg_parser()
    args_ = parser.parse_args()

    test_inference(args_)
