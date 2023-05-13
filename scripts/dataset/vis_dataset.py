import argparse
import random

import numpy as np
import torch
import trimesh
from tqdm import trange

from neural_contact_fields.data.tool_dataset import ToolDataset
from neural_contact_fields.utils import vedo_utils
from neural_contact_fields.utils.model_utils import load_dataset_from_config

from vedo import Plotter, Mesh, Points


def vis_dataset(dataset, base_mesh_fn: str = None, offset: int = 0):
    print("Dataset size: %d" % len(dataset))

    base_mesh = trimesh.load_mesh(base_mesh_fn) if base_mesh_fn is not None else None
    base_mesh.apply_translation([0.0, 0.0, 0.036])

    for data_idx in trange(offset, len(dataset)):
        data_dict = dataset[data_idx]

        partial_pc = data_dict["partial_pointcloud"]
        contact_patch = data_dict["contact_patch"]
        if type(dataset) == ToolDataset:
            mesh = dataset.get_example_mesh(data_idx)
        else:
            mesh = None

        # Visualize.
        plt = Plotter()
        vis_actors = [
            Points(partial_pc, c="black"),
            Points(contact_patch, c="red"),
            vedo_utils.draw_origin(),
        ]
        if mesh is not None:
            vis_actors.append(Mesh([mesh.vertices, mesh.faces], c="grey", alpha=0.5))
        if base_mesh is not None:
            vis_actors.append(Mesh([base_mesh.vertices, base_mesh.faces], c="grey", alpha=0.5))
        plt.at(0).show(*vis_actors)
        plt.interactive().close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize dataset.")
    parser.add_argument("dataset_cfg", type=str, help="Path to dataset config.")
    parser.add_argument("--mode", "-m", type=str, default="test", help="Which dataset split to use [train, val, test].")
    parser.add_argument('--offset', type=int, default=0, help='Offset to start from.')
    parser.add_argument("--base_mesh", type=str, default=None, help="Base mesh to use for visualization.")
    args = parser.parse_args()

    _, dataset_ = load_dataset_from_config(args.dataset_cfg, dataset_mode=args.mode)
    vis_dataset(dataset_, base_mesh_fn=args.base_mesh, offset=args.offset)
