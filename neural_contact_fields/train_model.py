import numpy as np
import os
import argparse
from tqdm import tqdm
import time
import torch
import torch.utils.data as data
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
    min_epochs = cfg['training']['min_epochs']
    max_epochs_without_improving = cfg['training']['max_epochs_without_improving']
    vis_dir = os.path.join(out_dir, 'vis')

    # Output + vis directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    logger = SummaryWriter(os.path.join(out_dir, 'logs'))

    # Dump config to output directory.
    utils.dump_cfg(os.path.join(out_dir, 'config.yaml'), cfg)

    # Create model:
    model = config.get_model(cfg, device=device)
    print(model)

    # Setup datasets.
    print('Loading train dataset:')
    train_dataset = config.get_dataset('train', cfg)
    print('Dataset size: %d' % len(train_dataset))
    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=cfg['training']['shuffle'],
        num_workers=16,
        drop_last=True,
        # pin_memory=True
    )
    print('Loading val dataset:')
    validation_dataset = config.get_dataset('val', cfg)
    val_dataloader = data.DataLoader(validation_dataset, batch_size=cfg['training']['val_batch_size'], shuffle=True,
                                     num_workers=32, drop_last=True)

    # For vis.
    # vis_dataloader = data.DataLoader(validation_dataset, batch_size=10, shuffle=True)
    # data_vis = next(iter(vis_dataloader))

    # Get optimizer (TODO: Parameterize?)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Get trainer.
    trainer = config.get_trainer(model, optimizer, cfg, logger, vis_dir, device=device)

    # Load model pretrain weights, if exists.
    pretrain_file = cfg['training'].get("pretrain_file", None)
    if pretrain_file is not None:
        print("Loading from pretraining file..")
        model_dict = {
            "model": model,
        }
        model_utils.load_model(model_dict, pretrain_file)

    # Load model + optimizer if exists.
    model_dict = {
        'model': model,
        'optimizer': optimizer,
    }
    model_file = os.path.join(out_dir, 'model.pt')
    load_dict = model_utils.load_model(model_dict, model_file)
    epoch_it = load_dict.get('epoch_it', -1)
    it = load_dict.get('it', -1)
    metric_val_best = load_dict.get('val_loss_best', np.inf)
    epoch_without_improving = 0

    # Training loop
    start_time = time.time()
    while True:
        epoch_it += 1

        if epoch_it > max_epochs or (epoch_without_improving > max_epochs_without_improving and epoch_it > min_epochs):
            print("%s Backing up and stopping training." % (
                "Reached max epochs." if epoch_it > max_epochs
                else "Went %d epochs without improving." % epoch_without_improving))
            save_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch_it': epoch_it,
                'it': it,
                'val_loss_best': metric_val_best
            }
            torch.save(save_dict, os.path.join(out_dir, 'model.pt'))
            break

        if epoch_it > 1:
            end_time = time.time()
            per_epoch_avg = (end_time - start_time) / (epoch_it - 1.0)
            # print("Avg per epoch time: ", per_epoch_avg) TODO: Save this information somewhere? Along with total.
            #  training time and runs information? Where should this go?

        for batch in train_dataloader:
            it += 1

            loss = trainer.train_step(batch, it)

            # Print output
            if print_every > 0 and (it % print_every) == 0:
                print('[Epoch %02d] it=%03d, loss=%.4f'
                      % (epoch_it, it, loss))

            # TODO: Bring back visualization?
            # if visualize_every > 0 and (it % visualize_every) == 0:
            #     print('Visualizing.')
            #     trainer.visualize(data_vis)

        # Validate after each batch.
        print('Validating.')
        val_dict = trainer.validation(val_dataloader, it)

        for k, v in val_dict.items():
            if v is not None:
                logger.add_scalar(k, v, epoch_it)

        val_loss = val_dict['val_loss']
        if val_loss < metric_val_best:
            epoch_without_improving = 0
            metric_val_best = val_loss
            print('Saving new best model. Loss=%03f' % metric_val_best)
            save_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch_it': epoch_it,
                'it': it,
                'val_loss_best': metric_val_best
            }
            torch.save(save_dict, os.path.join(out_dir, 'model_best.pt'))
        else:
            epoch_without_improving += 1

        # Backup.
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch_it': epoch_it,
            'it': it,
            'val_loss_best': metric_val_best
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
