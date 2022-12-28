import os
import argparse

from neural_contact_fields.data.tool_dataset import ToolDataset
import torch
import json

import mmint_utils as utils
import neural_contact_fields.config as config


def pretrain_model(config_file: str, cuda_id: int = 0, no_cuda: bool = False, verbose: bool = False,
                   config_args: dict = None):
    # Read config.
    cfg = utils.load_cfg(config_file)

    # If any customization is passed via command line - add in here.
    if config_args is not None:
        cfg = utils.combine_cfg(cfg, config_args)

    is_cuda = (torch.cuda.is_available() and not no_cuda)
    device = torch.device("cuda:%d" % cuda_id if is_cuda else "cpu")

    # Setup datasets.
    print('Loading pretrain dataset:')
    pretrain_dataset: ToolDataset = config.get_dataset('pretrain', cfg)
    print('Pretrain dataset size: %d' % len(pretrain_dataset))

    # Create model:
    print('Loading model:')
    model = config.get_model(cfg, device=device)
    print(model)

    # Get trainer.
    trainer = config.get_trainer(cfg, model, device=device)
    trainer.pretrain(pretrain_dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--cuda_id', type=int, default=0, help="Cuda device id to use.")
    parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Be verbose.')
    parser.set_defaults(verbose=False)
    parser.add_argument('--config_args', type=json.loads, default=None,
                        help='Config elements to overwrite. Use for easy hyperparameter search.')
    args = parser.parse_args()

    pretrain_model(args.config, args.cuda_id, args.no_cuda, args.verbose, args.config_args)
