import os
import argparse
import pdb

from neural_contact_fields.data.tool_dataset import ToolDataset
import torch
import torch.optim as optim
import json
from tensorboardX import SummaryWriter

import mmint_utils as utils
import neural_contact_fields.config as config
import neural_contact_fields.model_utils as model_utils


def train_model(config_file: str, cuda_id: int = 0, no_cuda: bool = False, verbose: bool = False,
                config_args: dict = None):
    # Read config.
    cfg = utils.load_cfg(config_file)

    # If any customization is passed via command line - add in here.
    if config_args is not None:
        cfg = utils.combine_cfg(cfg, config_args)

    is_cuda = (torch.cuda.is_available() and not no_cuda)
    device = torch.device("cuda:%d" % cuda_id if is_cuda else "cpu")

    # Shorthands:
    out_dir = cfg['training']['out_dir']
    lr = cfg['training']['learning_rate']
    print_every = cfg['training']['print_every']
    max_epochs = cfg['training']['epochs']
    vis_dir = os.path.join(out_dir, 'vis')

    # Output + vis directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    logger = SummaryWriter(os.path.join(out_dir, 'logs'))

    # Dump config to output directory.
    utils.dump_cfg(os.path.join(out_dir, 'config.yaml'), cfg)

    # Setup datasets.
    print('Loading train dataset:')
    train_dataset: ToolDataset = config.get_dataset('train', cfg)
    print('Dataset size: %d' % len(train_dataset))

    # Create model:
    model = config.get_model(cfg, train_dataset, device=device)
    print(model)

    # Get optimizer (TODO: Parameterize?)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Get trainer.
    trainer = config.get_trainer(model, optimizer, cfg, logger, vis_dir, device=device)

    # Load model pretrain weights, if exists.
    pretrain_file = cfg['training'].get("pretrain_file", None)
    if pretrain_file is not None:
        print("Loading pretrained model weights from local file: %s" % pretrain_file)
        pretrain_state_dict = torch.load(pretrain_file, map_location='cpu')

        object_module_keys = [key for key in pretrain_state_dict["model"].keys() if "object_module" in key]
        object_module_dict = {k: pretrain_state_dict["model"][k] for k in object_module_keys}

        model.load_state_dict(object_module_dict, strict=False)

        for param in model.object_module.parameters():
            param.requires_grad = False

    # Load model + optimizer if exists.
    model_dict = {
        'model': model,
        'optimizer': optimizer,
    }
    model_file = os.path.join(out_dir, 'model.pt')
    load_dict = model_utils.load_model(model_dict, model_file)
    epoch_it = load_dict.get('epoch_it', -1)
    it = load_dict.get('it', -1)

    # Training loop
    while True:
        epoch_it += 1

        if epoch_it > max_epochs:
            print("Backing up and stopping training. Reached max epochs.")
            save_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch_it': epoch_it,
                'it': it,
            }
            torch.save(save_dict, os.path.join(out_dir, 'model.pt'))
            break

        loss = None
        for trial_idx in range(train_dataset.num_trials):
            it += 1

            batch = train_dataset.get_all_points_for_trial_batch(-1, trial_idx)
            loss = trainer.train_step(batch, it)

        print('[Epoch %02d] it=%03d, loss=%.4f'
              % (epoch_it, it, loss))

        # Backup.
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch_it': epoch_it,
            'it': it,
        }
        torch.save(save_dict, os.path.join(out_dir, 'model.pt'))


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

    train_model(args.config, args.cuda_id, args.no_cuda, args.verbose, args.config_args)
