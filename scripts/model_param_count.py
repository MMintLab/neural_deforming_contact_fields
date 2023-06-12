import argparse

import torch

import mmint_utils as utils
import neural_contact_fields.config as config
from neural_contact_fields.data.tool_dataset import ToolDataset


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_param_count(config_file: str, cuda_id: int = 0, no_cuda: bool = False, config_args: dict = None):
    # Read config.
    cfg = utils.load_cfg(config_file)

    # If any customization is passed via command line - add in here.
    if config_args is not None:
        cfg = utils.combine_dict(cfg, config_args)

    is_cuda = (torch.cuda.is_available() and not no_cuda)
    device = torch.device("cuda:%d" % cuda_id if is_cuda else "cpu")

    # Setup datasets.
    train_dataset: ToolDataset = config.get_dataset('train', cfg, load_data=False, device=device)

    # Create model:
    print('Loading model:')
    model = config.get_model(cfg, train_dataset, device=device)
    # print(model)

    print("Number of parameters: %d" % count_parameters(model))
    print("Number of parameters (no codes): %d" %
          (count_parameters(model.object_model) + count_parameters(model.deformation_model) + count_parameters(
              model.contact_model) + count_parameters(model.wrench_encoder))
          )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get model parameter count.")
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()

    model_param_count(args.config)
