from collections import Callable

import numpy as np
import torch
import torch.optim as optim

import torch.nn as nn
from tqdm import trange


def inference_by_optimization(model: nn.Module, loss_fn: Callable, init_fn: Callable, latent_size: int,
                              num_latent: int, data_dict: dict,
                              inf_params=None, device: torch.device = None, verbose: bool = False):
    """
    Helper with basic inference by optimization structure. Repeatedly calls loss function with the specified
    data/loss function and updates latent inputs accordingly.

    Args:
    - model (nn.Module): network model
    - loss_fn (Callable): loss function. Should take in model, current latent, data dictionary, and device and return loss.
    - init_fn (Callable): initialization function. Should init the given embedding.
    - latent_size (int): specify latent space size.
    - num_examples (int): number of examples to run inference on.
    - data_dict (dict): data dictionary for example(s) we are inferring for.
    - inf_params (dict): inference hyper-parameters.
    - device (torch.device): pytorch device.
    - epsilon (float): convergence threshold.
    - verbose (bool): be verbose.
    """
    if inf_params is None:
        inf_params = {}
    model.eval()

    # Load inference hyper parameters.
    lr = inf_params.get("lr", 3e-2)
    num_steps = inf_params.get("iter_limit", 300)

    # Initialize latent code as noise.
    z_ = nn.Embedding(num_latent, latent_size, dtype=torch.float32).requires_grad_(True).to(device)
    init_fn(z_)  # Call provided init function.
    optimizer = optim.Adam(z_.parameters(), lr=lr)

    # Start optimization procedure.
    z = z_.weight

    iter_idx = 0
    if verbose:
        range_ = trange(num_steps)
    else:
        range_ = range(num_steps)
    for iter_idx in range_:
        optimizer.zero_grad()

        loss, _ = loss_fn(model, z, data_dict, device)

        loss.backward()
        optimizer.step()

        if verbose:
            range_.set_postfix(loss=loss.item())
    if verbose:
        range_.close()

    _, final_loss = loss_fn(model, z, data_dict, device)
    return z_, {"final_loss": final_loss, "iters": iter_idx + 1}
