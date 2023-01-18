import pdb

import mmint_utils
import torch
import os
import torch.nn as nn
import neural_contact_fields.config as config


def get_num_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_model(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def load_model(model_dict, model_file):
    """
    Load model dictionary elements from given model file.
    Returns dictionary with other elements in file (e.g., training
    iter info and val loss).

    Args:
    - model_dict (dict): dict of model key and pytorch objects to load.
    - model_file (str): path to saved model to load.
    """
    if os.path.exists(model_file):
        print('Loading checkpoint from local file: %s' % model_file)
        state_dict = torch.load(model_file, map_location='cpu')
        for k, v in model_dict.items():
            if k in state_dict:
                v.load_state_dict(state_dict[k])
            else:
                print('Warning: Could not find %s in checkpoint' % k)
        load_dict = {k: v for k, v in state_dict.items()
                     if k not in model_dict}
    else:
        print("Couldn't find model file at ", model_file)
        load_dict = dict()

    return load_dict


def load_dataset_from_config(dataset_config, dataset_mode="test"):
    # Load dataset.
    dataset_cfg = mmint_utils.load_cfg(dataset_config)
    dataset = config.get_dataset(dataset_mode, dataset_cfg)
    return dataset


def load_model_from_config(model_config, dataset, model_file="model_best.pt", cuda_id=0, no_cuda=False):
    # Read in configuration files.
    model_cfg = mmint_utils.load_cfg(model_config)

    # Setup cuda.
    is_cuda = (torch.cuda.is_available() and not no_cuda)
    cuda_device = torch.device("cuda:%d" % cuda_id if is_cuda else "cpu")

    # Create model:
    model = config.get_model(model_cfg, dataset, device=cuda_device)

    # Load model from file.
    model_dict = {
        'model': model,
    }
    model_file = os.path.join(model_cfg['training']['out_dir'], model_file)
    _ = load_model(model_dict, model_file)
    return model_cfg, model, cuda_device


def load_model_and_dataset(model_config, dataset_config=None, dataset_mode="test", model_file='model_best.pt',
                           cuda_id=0, no_cuda=False):
    if dataset_config is None:
        dataset_config = model_config
    dataset = load_dataset_from_config(dataset_config, dataset_mode)

    # Because some models depend on their train dataset, we load it additionally.
    train_dataset = load_dataset_from_config(model_config, dataset_mode="train")
    model_cfg, model, cuda_device = load_model_from_config(model_config, train_dataset, model_file=model_file,
                                                           cuda_id=cuda_id, no_cuda=no_cuda)
    return model_cfg, model, dataset, cuda_device
