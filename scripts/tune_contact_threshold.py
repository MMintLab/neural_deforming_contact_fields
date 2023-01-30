import argparse
import os
import random

import mmint_utils
import numpy as np
import pytorch3d.loss
import torch
import tqdm
from matplotlib import pyplot as plt
from neural_contact_fields import config
from neural_contact_fields.utils.model_utils import load_model_and_dataset
from neural_contact_fields.utils.results_utils import load_pred_results, write_results, load_gt_results
from tqdm import trange


def tune_contact_threshold(model_cfg, model, val_dataset_cfg, val_dataset, out_dir: str = None, device=None,
                           load_dict: dict = None, vis: bool = False):
    model.eval()
    dataset_size = len(val_dataset)

    # Load generator.
    generator = config.get_generator(model_cfg, model, {}, device)

    print("Loading/Generating contact meshes for patch evaluation...")

    # Load ground truth.
    _, _, gt_contact_patches, _, _, _ = load_gt_results(val_dataset, val_dataset_cfg["data"]["val"]["dataset_dir"],
                                                        dataset_size, device)

    # If out directory provided, check if data already exists.
    if out_dir is not None and os.path.exists(out_dir):
        pred_meshes, _, _, _ = load_pred_results(out_dir, dataset_size, device)

        # Also load latent values, if present.
        pred_latents = []
        for idx in range(dataset_size):
            latent_fn = os.path.join(out_dir, "latent_%d.pkl.gzip" % idx)
            if os.path.exists(latent_fn):
                pred_latents.append(mmint_utils.load_gzip_pickle(latent_fn))
            else:
                pred_latents.append(None)
    elif out_dir is not None and not os.path.exists(out_dir):
        mmint_utils.make_dir(out_dir)
        pred_meshes = [None] * dataset_size
        pred_latents = [None] * dataset_size
    else:
        pred_meshes = [None] * dataset_size
        pred_latents = [None] * dataset_size

    # Generate meshes/latents.
    for idx in trange(dataset_size):
        data_dict = val_dataset[idx]

        # If data has already been generated, skip generation step.
        if pred_meshes[idx] is not None:
            continue

        # Generate contact labels for example.
        pred_mesh, metadata = generator.generate_mesh(data_dict, {})
        pred_meshes[idx] = pred_mesh
        pred_latents[idx] = metadata["latent"]

        # Save generated contact labels to file.
        if out_dir is not None:
            write_results(out_dir, pred_mesh, None, None, None, idx)
            latent_fn = os.path.join(out_dir, "latent_%d.pkl.gzip" % idx)
            mmint_utils.save_gzip_pickle(metadata["latent"], latent_fn)

    # Search through contact thresholds for best performance.
    print("Searching thresholds...")
    thresholds = np.arange(0.0, 1.000000001, 0.01)
    chamfer_dists = []

    for threshold in tqdm.tqdm(thresholds):
        # Set threshold.
        generator.contact_threshold = threshold

        threshold_chamfer_dists = []
        for idx in range(dataset_size):
            data_dict = val_dataset[idx]

            pred_cp, _ = generator.generate_contact_patch(data_dict,
                                                          {"mesh": pred_meshes[idx], "latent": pred_latents[idx]})

            chamfer_dist, _ = pytorch3d.loss.chamfer_distance(torch.from_numpy(pred_cp).float().to(device).unsqueeze(0),
                                                              gt_contact_patches[idx].unsqueeze(0))
            threshold_chamfer_dists.append(chamfer_dist.item())
        chamfer_dists.append(np.mean(threshold_chamfer_dists))

    if vis:
        plt.plot(thresholds, chamfer_dists)
        plt.xlim(0.0, 1.0)
        plt.xlabel("Binary Thresholds")
        plt.ylabel("Chamfer Distance")
        plt.show()

    # Determine best threshold using scores.
    best_idx = np.argmin(chamfer_dists)
    best_threshold = thresholds[best_idx]
    best_score = chamfer_dists[best_idx]
    print("Best threshold: %f. Score: %f." % (best_threshold, best_score))

    # Save threshold to model file. This allows it to be easily loaded at test time.
    model_dict = {
        "model": model.state_dict(),
        "generation": {
            "contact_threshold": best_threshold,
        },
    }
    model_dict.update(load_dict)
    model_dir = model_cfg["training"]["out_dir"]
    torch.save(model_dict, os.path.join(model_dir, "model_threshold.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Model/data config file.")
    parser.add_argument("val_dataset_config", type=str, help="Validation dataset config to use.")
    parser.add_argument("--model_file", "-f", type=str, default="model.pt", help="Which model save file to use.")
    parser.add_argument("--out_dir", "-o", type=str, default=None,
                        help="Optional out dir to load/save generated labels used.")
    parser.add_argument("--vis", "-v", action='store_true', help="Visualize results.")
    parser.set_defaults(vis=False)
    args = parser.parse_args()

    # Seed for repeatability.
    torch.manual_seed(10)
    np.random.seed(10)
    random.seed(10)

    model_cfg_, model_, val_dataset_, device_, load_dict_ = load_model_and_dataset(
        args.config, dataset_config=args.val_dataset_config, dataset_mode="val", model_file=args.model_file
    )
    val_dataset_cfg_ = mmint_utils.load_cfg(args.val_dataset_config)

    tune_contact_threshold(model_cfg_, model_, val_dataset_cfg_, val_dataset_, args.out_dir, device_, load_dict_,
                           args.vis)
