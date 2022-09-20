import numpy as np
import torch
from tqdm import trange


def points_inference(model, n: int = 256, max_batch: int = 40 ** 3, device=None):
    model.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (n - 1)

    overall_index = torch.arange(0, n ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(n ** 3, 5)  # x, y, z, sdf, contact

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % n
    samples[:, 1] = (overall_index.long() / n) % n
    samples[:, 0] = ((overall_index.long() / n) / n) % n

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = n ** 3

    head = 0
    num_iters = int(np.ceil(num_samples / max_batch))
    # while head < num_samples:
    for iter_idx in trange(num_iters):
        sample_subset = samples[head: min(head + max_batch, num_samples), 0:3].to(device)

        with torch.no_grad():
            sdf_pred, _, contact_pred = model(sample_subset)

        samples[head: min(head + max_batch, num_samples), 3] = sdf_pred.squeeze(-1).cpu()
        samples[head: min(head + max_batch, num_samples), 4] = contact_pred.squeeze(-1).cpu()

        head += max_batch

    return samples
