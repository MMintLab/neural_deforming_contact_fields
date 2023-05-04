import os.path
import time

import mmint_utils
import numpy as np
import torch
import yaml
from neural_contact_fields import config
from neural_contact_fields.utils.args_utils import get_model_dataset_arg_parser, load_model_dataset_from_args
from neural_contact_fields.utils.model_utils import load_generation_cfg
from neural_contact_fields.utils.results_utils import write_results
from tqdm import trange
import random


def generate_example(generator, data_dict):
    # TODO: Move to different file.
    metadata = {}
    mesh = pointcloud = contact_patch = contact_labels = None

    # Generate latent.
    latent, latent_metadata = generator.generate_latent(data_dict)
    metadata["latent"] = latent
    metadata = mmint_utils.combine_dict(metadata, latent_metadata)

    if generator.generates_mesh:
        mesh, metadata_mesh = generator.generate_mesh(data_dict, metadata)
        metadata = mmint_utils.combine_dict(metadata, metadata_mesh)

    if generator.generates_pointcloud:
        pointcloud, metadata_pc = generator.generate_pointcloud(data_dict, metadata)
        metadata = mmint_utils.combine_dict(metadata, metadata_pc)

    if generator.generates_contact_patch:
        contact_patch, metadata_cp = generator.generate_contact_patch(data_dict, metadata)
        metadata = mmint_utils.combine_dict(metadata, metadata_cp)

    if generator.generates_contact_labels:
        contact_labels, metadata_cl = generator.generate_contact_labels(data_dict, metadata)
        metadata = mmint_utils.combine_dict(metadata, metadata_cl)

    return {
        "mesh": mesh, "pointcloud": pointcloud, "contact_patch": contact_patch, "contact_labels": contact_labels,
        "metadata": metadata
    }


def generate(model_cfg, model, model_file, dataset, device, out_dir, gen_args: dict, offset: int):
    model.eval()

    # Load generate cfg, if present.
    generation_cfg = load_generation_cfg(model_cfg, model_file)
    if gen_args is not None:
        generation_cfg.update(gen_args)

    # Load generator.
    generator = config.get_generator(model_cfg, model, generation_cfg, device)

    # Create output directory.
    if out_dir is not None:
        mmint_utils.make_dir(out_dir)

    # Dump any generation arguments to out directory.
    mmint_utils.dump_cfg(os.path.join(out_dir, "metadata.yaml"), generation_cfg)

    # Go through dataset and generate!
    for idx in trange(offset, len(dataset)):
        data_dict = dataset[idx]
        gen_dict = generate_example(generator, data_dict)

        write_results(out_dir, gen_dict, idx)


if __name__ == '__main__':
    parser = get_model_dataset_arg_parser()
    parser.add_argument("--out", "-o", type=str, help="Optional out directory to write generated results to.")
    # TODO: Add visualization?
    parser.add_argument("--gen_args", type=yaml.safe_load, default=None, help="Generation args.")
    parser.add_argument("--offset", type=int, default=0, help="Offset to add to config indices.")
    args = parser.parse_args()

    # Seed for repeatability.
    torch.manual_seed(10)
    np.random.seed(10)
    random.seed(10)

    model_cfg_, model_, dataset_, device_ = load_model_dataset_from_args(args)
    generate(model_cfg_, model_, args.model_file, dataset_, device_, args.out, args.gen_args, args.offset)
