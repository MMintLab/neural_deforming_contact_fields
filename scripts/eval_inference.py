import os
import random
import time

import numpy as np
import torch
import yaml
from tqdm import trange, tqdm

import mmint_utils
from neural_contact_fields import config
from neural_contact_fields.utils.args_utils import get_model_dataset_arg_parser, load_model_dataset_from_args
from neural_contact_fields.utils.model_utils import load_generation_cfg
from neural_contact_fields.utils.results_utils import write_results


def eval_inference(model_cfg, model, model_file, dataset, device, out_dir, gen_args: dict, n_repeat: int = 1):
    """
    To account for stochasticity in inference procedure, we repeat runs multiple times.
    """
    model.eval()

    # Load generate cfg, if present.
    generation_cfg = load_generation_cfg(model_cfg, model_file)
    if gen_args is not None:
        generation_cfg.update(gen_args)

    # Create generator.
    generator = config.get_generator(model_cfg, model, generation_cfg, device)

    # Create output directory.
    if out_dir is not None:
        mmint_utils.make_dir(out_dir)

    # Dump any generation arguments to out directory.
    mmint_utils.dump_cfg(os.path.join(out_dir, "metadata.yaml"), generation_cfg)

    with tqdm(total=len(dataset) * n_repeat) as pbar:
        for run_idx in range(n_repeat):
            run_dir = os.path.join(out_dir, f"run_{run_idx}")
            mmint_utils.make_dir(run_dir)

            # Generate results.
            for idx in range(len(dataset)):
                data_dict = dataset[idx]

                # Generate latent.
                start_time = time.time()
                latent = generator.generate_latent(data_dict)
                end_time = time.time()
                latent_gen_time = end_time - start_time

                contact_labels_dict, _ = generator.generate_contact_labels(data_dict, {"latent": latent})
                iou_labels_dict, _ = generator.generate_iou_labels(data_dict, {"latent": latent})

                write_results(run_dir, None, None, None, contact_labels_dict, iou_labels_dict, idx,
                              {"latent_gen_time": latent_gen_time, "latent": latent})

                pbar.update(1)


if __name__ == '__main__':
    parser = get_model_dataset_arg_parser()
    parser.add_argument("--out", "-o", type=str, help="Optional out directory to write generated results to.")
    parser.add_argument("--gen_args", type=yaml.safe_load, default=None, help="Generation args.")
    parser.add_argument("-n", "--n_repeat", type=int, default=1, help="Number of times to repeat inference.")
    args = parser.parse_args()

    # Seed for repeatability.
    torch.manual_seed(10)
    np.random.seed(10)
    random.seed(10)

    model_cfg_, model_, dataset_, device_ = load_model_dataset_from_args(args)
    eval_inference(model_cfg_, model_, args.model_file, dataset_, device_, args.out, args.gen_args, args.n_repeat)
