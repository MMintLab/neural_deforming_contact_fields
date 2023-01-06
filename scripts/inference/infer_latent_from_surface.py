import mmint_utils
import torch
from neural_contact_fields.data.tool_dataset import ToolDataset
from neural_contact_fields.inference import infer_latent_from_surface
from neural_contact_fields.utils.model_utils import load_model_and_dataset
import argparse
from scripts.train.vis_prediction_vs_dataset import vis_prediction_vs_dataset


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

    for trial_idx in range(len(dataset)):
        trial_dict = dataset[trial_idx]
        latent_code, pred_dict = infer_latent_from_surface(model, trial_dict, {}, device=device)

        results_dict = {
            "gt": numpy_dict(trial_dict),
            "pred": numpy_dict(pred_dict)
        }
        vis_prediction_vs_dataset(results_dict)


if __name__ == '__main__':
    parser = get_model_dataset_arg_parser()
    args_ = parser.parse_args()

    test_inference(args_)
