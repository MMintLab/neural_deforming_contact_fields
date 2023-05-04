import argparse

from neural_contact_fields.utils.model_utils import load_model_and_dataset


def get_model_dataset_arg_parser():
    """
    Argument parser for common model + dataset arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Model/data config file.")
    parser.add_argument("--dataset_config", "-d", type=str, default=None, help="Optional dataset config to use.")
    parser.add_argument("--mode", "-m", type=str, default="test", help="Which dataset split to use [train, val, test].")
    parser.add_argument("--model_file", "-f", type=str, default="model.pt", help="Which model save file to use.")
    parser.add_argument('--cuda_id', type=int, default=0, help="Cuda device id to use.")
    parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')
    return parser


def load_model_dataset_from_args(args, load_data: bool = True):
    """
    Load model and dataset from arguments object.
    """
    model_cfg, model, dataset, device, load_dict = load_model_and_dataset(
        args.config,
        dataset_config=args.dataset_config,
        dataset_mode=args.mode,
        model_file=args.model_file,
        load_data=load_data,
        no_cuda=args.no_cuda,
        cuda_id=args.cuda_id
    )

    model.eval()
    return model_cfg, model, dataset, device
