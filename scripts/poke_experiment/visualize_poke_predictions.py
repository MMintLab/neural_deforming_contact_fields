import os
import argparse

import numpy as np
import trimesh
from vedo import Plotter, Mesh, Points

from neural_contact_fields.utils import utils


def visualize_poke_predictions(data_dir, gen_dir, n):
    # Load partial point clouds from setup.
    obj_motioncam = utils.load_pointcloud(os.path.join(data_dir, "obj_motioncam.ply"))[:, :3]
    obj_photoneo = utils.load_pointcloud(os.path.join(data_dir, "obj_photoneo.ply"))[:, :3]
    obj_combined = np.concatenate([obj_motioncam, obj_photoneo], axis=0)

    # For each step load predicted contact patch and mesh.
    for step_idx in range(n):
        # Load predicted contact patch.
        pred_contact_patch = utils.load_pointcloud(os.path.join(gen_dir, "contact_patch_w_%d.ply" % step_idx))
        # Load predicted mesh.
        pred_mesh = trimesh.load(os.path.join(gen_dir, "mesh_w_%d.obj" % step_idx))

        # Visualize.
        plt = Plotter(shape=(1, 2))
        plt.at(0).show(
            Mesh([pred_mesh.vertices, pred_mesh.faces], c="gold", alpha=1.0),
            Points(obj_combined, c="black")
        )
        plt.at(1).show(
            Points(pred_contact_patch[:, :3], c="r"),
            Points(obj_combined, c="black")
        )
        plt.interactive().close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize poke predictions.")
    parser.add_argument("data_dir", type=str, help="Original data directory.")
    parser.add_argument("gen_dir", type=str, help="Dir where generations were output.")
    parser.add_argument("n", type=int, help="Num examples in dir.")
    args = parser.parse_args()

    visualize_poke_predictions(args.data_dir, args.gen_dir, args.n)
