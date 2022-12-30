import pdb
from collections import defaultdict, OrderedDict

import numpy as np
import torch
from tqdm import trange
import torch.optim as optim

import torch.nn.functional as F


def points_inference_latent(model, latent, query_points, max_batch: int = 40 ** 3, device=None):
    model.eval()

    num_samples = query_points.shape[0]

    head = 0
    num_iters = int(np.ceil(num_samples / max_batch))
    pred_dict_all = defaultdict(list)

    for iter_idx in trange(num_iters):
        sample_subset = query_points[head: min(head + max_batch, num_samples)]
        latent_in = latent.repeat(sample_subset.shape[0], 1)

        with torch.no_grad():
            pred_dict = model.latent_forward(latent_in, sample_subset)

        for k, v in pred_dict.items():
            pred_dict_all[k].append(v)

        head += max_batch

    pred_dict_final = dict()
    for k, v in pred_dict_all.items():
        pred_dict_final[k] = torch.cat(v, dim=0)

    return pred_dict_final


def infer_latent(model, trial_dict, max_batch: int = 40 ** 3, device=None):
    model.eval()

    query_points = torch.from_numpy(trial_dict["query_point"]).to(device).float()
    sdf = torch.from_numpy(trial_dict["sdf"]).float().to(device)
    surface_query_points = sdf == 0.0
    in_contact = torch.from_numpy(trial_dict["in_contact"]).to(device) > 0.5
    in_contact_float = torch.clone(in_contact).float()
    force = torch.from_numpy(trial_dict["force"]).float().to(device)
    num_samples = query_points.shape[0]

    # Initialize latent code as noise.
    latent_code = torch.zeros([1, 64], requires_grad=True, dtype=torch.float32, device=device)
    torch.nn.init.normal_(latent_code, mean=0.0, std=0.1)
    optimizer = optim.Adam([latent_code], lr=1e-3)

    for _ in trange(10000):
        optimizer.zero_grad()

        head = 0
        num_iters = int(np.ceil(num_samples / max_batch))
        pred_dict_all = defaultdict(list)
        for iter_idx in range(num_iters):
            sample_subset = query_points[head: min(head + max_batch, num_samples)]
            latent_in = latent_code.repeat(sample_subset.shape[0], 1)

            pred_dict = model.latent_forward(latent_in, sample_subset)

            for k, v in pred_dict.items():
                pred_dict_all[k].append(v)

            head += max_batch

        # Apply losses.
        loss_dict = dict()

        # We apply the SDF loss to every point in space.
        sdf_loss = F.l1_loss(pred_dict["sdf"], sdf, reduction="mean")
        loss_dict["sdf_loss"] = sdf_loss

        # Apply L2 loss on deformation.
        def_loss = torch.linalg.norm(pred_dict["delta_coords"]).mean()
        loss_dict["def_loss"] = def_loss

        # Next, for all points *on the surface* we predict the contact probability.
        if surface_query_points.sum() > 0:
            contact_loss = F.binary_cross_entropy_with_logits(pred_dict["contact_logits"][surface_query_points],
                                                              in_contact_float[surface_query_points], reduction="mean")
            loss_dict["contact_loss"] = contact_loss
        else:
            loss_dict["contact_loss"] = 0.0

        # Finally, for all points *in contact* we predict the contact forces.
        if in_contact.sum() > 0:
            force_loss = F.mse_loss(pred_dict["contact_force"][in_contact], force[in_contact], reduction="mean")
            loss_dict["force_loss"] = force_loss
        else:
            loss_dict["force_loss"] = 0.0

        # Combined losses. TODO: Parameterize weights.
        loss = sdf_loss + \
               (0.1 * loss_dict["contact_loss"]) + \
               (1.0 * loss_dict["force_loss"])

        loss.backward()
        optimizer.step()

    return latent_code


def infer_latent_from_surface(model, trial_dict, max_batch: int = 40 ** 3, device=None):
    model.eval()

    query_points = torch.from_numpy(trial_dict["query_point"]).to(device).float()
    sdf = torch.from_numpy(trial_dict["sdf"]).float().to(device)
    surface_query_points = sdf == 0.0
    num_samples = surface_query_points.shape[0]
    query_points = query_points[surface_query_points]

    # Initialize latent code as noise.
    latent_code = torch.zeros([1, 64], requires_grad=True, dtype=torch.float32, device=device)
    torch.nn.init.normal_(latent_code, mean=0.0, std=0.1)
    optimizer = optim.Adam([latent_code], lr=1e-3)

    for it in trange(100000):
        optimizer.zero_grad()

        head = 0
        num_iters = int(np.ceil(num_samples / max_batch))
        pred_dict_all = defaultdict(list)
        for iter_idx in range(num_iters):
            sample_subset = query_points[head: min(head + max_batch, num_samples)]
            latent_in = latent_code.repeat(sample_subset.shape[0], 1)

            pred_dict = model.latent_forward(latent_in, sample_subset)

            for k, v in pred_dict.items():
                pred_dict_all[k].append(v)

            head += max_batch

        # Apply surface loss.
        loss = torch.sum(torch.abs(pred_dict["sdf"]))

        if (it % 1000) == 0:
            print("Step %d: %f" % (it, loss))

        loss.backward()
        optimizer.step()

    return latent_code


def points_inference(model, object_code, trial_code, trial_dict, max_batch: int = 40 ** 3, device=None):
    model.eval()
    object_index = trial_dict["object_idx"]
    trial_index = trial_dict["trial_idx"]

    # Encode object idx/trial idx.
    object_idx_tensor = torch.tensor(object_index, device=device)
    trial_idx_tensor = torch.tensor(trial_index, device=device)
    z_object = object_code(object_idx_tensor).float()
    z_trial = trial_code(trial_idx_tensor).float()

    # Get query points to sample.
    query_points = torch.from_numpy(trial_dict["query_point"]).to(device).float()
    num_samples = query_points.shape[0]

    head = 0
    num_iters = int(np.ceil(num_samples / max_batch))
    pred_dict_all = defaultdict(list)

    for iter_idx in trange(num_iters):
        sample_subset = query_points[head: min(head + max_batch, num_samples)]
        trial_indices = torch.zeros(sample_subset.shape[0], dtype=torch.long).to(device) + trial_index

        with torch.no_grad():
            pred_dict = model.forward(sample_subset, z_trial, z_object)

        for k, v in pred_dict.items():
            pred_dict_all[k].append(v)

        head += max_batch

    pred_dict_final = dict()
    for k, v in pred_dict_all.items():
        if type(v[0]) is torch.Tensor:
            pred_dict_final[k] = torch.cat(v, dim=0)

    return pred_dict_final
