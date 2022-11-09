from collections import defaultdict

import numpy as np
import torch
from tqdm import trange


def points_inference(model, trial_dict, max_batch: int = 40 ** 3, device=None):
    model.eval()

    object_index = trial_dict["object_idx"]
    trial_index = trial_dict["trial_idx"]
    query_points = torch.from_numpy(trial_dict["query_point"]).to(device).float()
    num_samples = query_points.shape[0]

    head = 0
    num_iters = int(np.ceil(num_samples / max_batch))
    pred_dict_all = defaultdict(list)

    for iter_idx in trange(num_iters):
        sample_subset = query_points[head: min(head + max_batch, num_samples)]
        trial_indices = torch.zeros(sample_subset.shape[0], dtype=torch.long).to(device) + trial_index

        with torch.no_grad():
            pred_dict = model(trial_indices, sample_subset)

        for k, v in pred_dict.items():
            pred_dict_all[k].append(v)

        head += max_batch

    pred_dict_final = dict()
    for k, v in pred_dict_all.items():
        pred_dict_final[k] = torch.cat(v, dim=0)

    return pred_dict_final
