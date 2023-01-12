import argparse

from neural_contact_fields.data.tool_dataset import ToolDataset
from neural_contact_fields.neural_contact_field.models.virdo_ncf import VirdoNCF
from neural_contact_fields.utils.model_utils import load_model_and_dataset
import torch


def get_model_dataset_arg_parser():
    """
    Argument parser for common model + dataset arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Model/data config file.")
    # parser.add_argument("--out", "-o", type=str, default=None, help="Directory to write results to.")
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


def examine_trained_embeddings(args):
    vis = args.vis
    dataset: ToolDataset
    model: VirdoNCF
    model_cfg, model, dataset, device = load_model_dataset_from_args(args)

    # Pull out trial embedding.
    trial_embedding: torch.nn.Embedding = model.trial_code

    embedding_mean = torch.mean(trial_embedding.weight, dim=0)
    embedding_std = torch.std(trial_embedding.weight, dim=0)

    print("Mean embedding: " + str(embedding_mean.detach().cpu().numpy()))
    print("Std embedding: " + str(embedding_std.detach().cpu().numpy()))


if __name__ == '__main__':
    parser_ = get_model_dataset_arg_parser()
    args_ = parser_.parse_args()

    examine_trained_embeddings(args_)
