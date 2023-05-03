import argparse
import os
import random

import numpy as np
import torch
import yaml
from tqdm import trange

import mmint_utils
from neural_contact_fields import config
from neural_contact_fields.utils.args_utils import get_model_dataset_arg_parser, load_model_dataset_from_args
from neural_contact_fields.utils.model_utils import load_generation_cfg
from neural_contact_fields.utils.results_utils import write_nominal_results


def generate_nominal(model_cfg, model, model_file, dataset, device, out_dir, gen_args: dict, offset: int):
    model.eval()

    # Load generate cfg, if present.
    generation_cfg = load_generation_cfg(model_cfg, model_file)
    if gen_args is not None:
        generation_cfg.update(gen_args)

    # Load generator.
    generator = config.get_generator(model_cfg, model, generation_cfg, device)

    # Assert that the generator is a nominal generator.
    assert generator.generates_nominal_mesh

    # Create output directory.
    if out_dir is not None:
        mmint_utils.make_dir(out_dir)

    # Dump any generation arguments to out directory.
    mmint_utils.dump_cfg(os.path.join(out_dir, "metadata.yaml"), generation_cfg)

    num_objects = dataset.get_num_objects()

    # Go through dataset and generate nominal!
    for nominal_idx in trange(offset, num_objects):
        data_dict = {
            "object_idx": np.array([nominal_idx]),
        }
        metadata = {}

        nominal_mesh, metadata_nominal_mesh = generator.generate_nominal_mesh(data_dict, metadata)
        write_nominal_results(out_dir, nominal_mesh, nominal_idx, metadata_nominal_mesh)


if __name__ == '__main__':
    parser = get_model_dataset_arg_parser()
    parser.add_argument("--out", "-o", type=str, help="Optional out directory to write generated results to.")
    parser.add_argument("--gen_args", type=yaml.safe_load, default=None, help="Generation args.")
    parser.add_argument("--offset", type=int, default=0, help="Offset to add to config indices.")
    args = parser.parse_args()

    # Seed for repeatability.
    torch.manual_seed(10)
    np.random.seed(10)
    random.seed(10)

    model_cfg_, model_, dataset_, device_ = load_model_dataset_from_args(args, load_data=False)
    generate_nominal(model_cfg_, model_, args.model_file, dataset_, device_, args.out, args.gen_args, args.offset)
