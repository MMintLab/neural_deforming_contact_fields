import argparse
import os
from collections import defaultdict

import numpy as np
from vedo import Plotter, Points

from neural_contact_fields.utils import utils


def visualize_real_variations(source_dir: str, gen_base_dir: str, N: int = 10):
    # Load the setup pointclouds to visualize the "gt".
    setup_motioncam_pc = utils.load_pointcloud(os.path.join(source_dir, "setup_motioncam.ply"))
    setup_photoneo_pc = utils.load_pointcloud(os.path.join(source_dir, "setup_photoneo.ply"))

    # Variations dirs.
    var_dirs = os.listdir(gen_base_dir)
    num_vars = len(var_dirs)

    # For each example, load the contact patch.
    contact_patches = defaultdict(list)
    for i in range(N):
        for j, var_dir in enumerate(var_dirs):
            contact_patches[j].append(
                utils.load_pointcloud(os.path.join(gen_base_dir, var_dir, "contact_patch_w_%d.ply" % i)))

    # Visualize the contact patches.
    plt = Plotter(shape=[2, int(np.ceil(num_vars / 2.0))])
    for j in range(num_vars):
        plt.at(j).show(
            Points(setup_motioncam_pc, c="black"), Points(setup_photoneo_pc, c="black"),
            *[Points(contact_patches[j][i][:, :3], c="red") for i in range(N)],
            var_dirs[j],
        )
    plt.interactive().close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize real variations.")
    parser.add_argument("source_dir", type=str, help="Source data dir.")
    parser.add_argument("gen_base_dir", type=str, help="Base dir where variations are stored.")
    parser.add_argument("-n", "--num", type=int, default=10, help="Num examples to visualize.")
    args = parser.parse_args()

    visualize_real_variations(args.source_dir, args.gen_base_dir, args.num)
