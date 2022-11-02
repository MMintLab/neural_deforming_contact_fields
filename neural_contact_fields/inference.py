import numpy as np
import torch
from tqdm import trange
import mmint_utils


def points_inference(model, n: int = 256, max_batch: int = 40 ** 3, device=None):
    model.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    # voxel_origin = [-1, -1, -1]
    # voxel_size = 2.0 / (n - 1)

    index = 2
    example_dict = mmint_utils.load_gzip_pickle(
        "/home/markvdm/Documents/IsaacGym/ncf_envs/out/test_10_26_22/out_%d.pkl.gzip" % index)
    query_points = example_dict["query_points"]
    num_samples = query_points.shape[0]

    samples = torch.zeros(num_samples, 8, device=device)  # x, y, z, sdf, contact, fx, fy, fz
    samples[:, :3] = torch.from_numpy(query_points).to(device)

    head = 0
    num_iters = int(np.ceil(num_samples / max_batch))
    # while head < num_samples:
    for iter_idx in trange(num_iters):
        sample_subset = samples[head: min(head + max_batch, num_samples), 0:3].to(device)
        trial_indices = torch.zeros(sample_subset.shape[0], dtype=torch.long).to(device) + index

        with torch.no_grad():
            sdf_pred, _, contact_pred, contact_force = model(trial_indices, sample_subset)

        samples[head: min(head + max_batch, num_samples), 3] = sdf_pred.squeeze(-1)
        samples[head: min(head + max_batch, num_samples), 4] = contact_pred.squeeze(-1)
        samples[head: min(head + max_batch, num_samples), 5:] = contact_force

        head += max_batch

    mmint_utils.save_gzip_pickle(samples.cpu().numpy(), "test_inference_%d.pkl.gzip" % index)
    return samples.cpu().numpy()
