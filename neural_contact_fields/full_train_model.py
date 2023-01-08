import argparse
import json

from neural_contact_fields.pretrain_model import pretrain_model
from neural_contact_fields.train_model import train_model

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

    # Pretrain model.
    pretrain_model(args.config, args.cuda_id, args.no_cuda, args.verbose, args.config_args)

    # Train model.
    train_model(args.config, args.cuda_id, args.no_cuda, args.verbose, args.config_args)
