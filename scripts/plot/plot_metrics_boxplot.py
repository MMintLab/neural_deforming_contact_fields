import argparse
import os
from collections import defaultdict

from matplotlib import pyplot as plt

import mmint_utils
import numpy as np


def perf_to_csv(dirs, names=None, exp_name=None):
    if names is None:
        names = [os.path.basename(os.path.normpath(d)) for d in dirs]

    # TODO: Way to generalize this?
    env_ids = [[0, 1, 2]]
    env_names = ["all"]
    # metrics = ["patch_chamfer_distance", "chamfer_distance", "iou"]
    metrics = ["patch_chamfer_distance"]

    for env_id_set, env_name in zip(env_ids, env_names):
        res_dict = dict()
        for directory, name in zip(dirs, names):
            run_fn = os.path.join(directory, "metrics.pkl.gzip")
            run_dict = mmint_utils.load_gzip_pickle(run_fn)
            run_dict = [d for d in run_dict if d["metadata"]["env_id"] in env_id_set]

            # Collect result on each metric for each example.
            run_res_dict = defaultdict(list)
            for example_idx, example_run_dict in enumerate(run_dict):
                example_metrics_dict = example_run_dict["metrics"]

                for metric in metrics:
                    if metric in example_metrics_dict:
                        run_res_dict[metric].append(example_metrics_dict[metric])

            res_dict[name] = run_res_dict

        # Build boxplot data.
        fig, axs = plt.subplots(1, 2)
        axs[0] = plt.subplot(121)
        axs[1] = plt.subplot(122)
        boxplot_data = []
        for name_idx, name in enumerate(names):
            boxplot_data.append(res_dict[name]["patch_chamfer_distance"])
            axs[0].scatter(name_idx + 1, np.mean(res_dict[name]["patch_chamfer_distance"]), c="black", marker="*")
            axs[1].scatter(name_idx + 1, np.mean(res_dict[name]["patch_chamfer_distance"]), c="black", marker="*")
        axs[0].boxplot(boxplot_data, vert="True", widths=[0.8] * len(names))
        axs[1].boxplot(boxplot_data, vert="True", widths=[0.8] * len(names))
        axs[0].set_xticks(np.arange(1, len(names) + 1), labels=names)
        axs[1].set_xticks(np.arange(1, len(names) + 1), labels=names)
        axs[0].set_ylabel("Patch Chamfer Distance")
        # axs[1].set_ylabel("Patch Chamfer Distance")
        zoom_upper = 100
        axs[0].axhline(y=-3, color="blue", linestyle="--", lw=1.0)
        axs[0].axhline(y=zoom_upper + 3, color="blue", linestyle="--", lw=1.0)
        axs[1].set_ylim([-3, zoom_upper + 3])
        if exp_name is not None:
            fig.suptitle("%s" % exp_name)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Consolidate performance metrics.")
    parser.add_argument("dirs", type=str, nargs="+", help="Directories to consolidate.")
    parser.add_argument("-n", "--names", type=str, nargs="+", default=None, help="Names for each directory.")
    parser.add_argument("-e", "--exp_name", type=str, default=None, help="Experiment name.")
    args = parser.parse_args()

    perf_to_csv(args.dirs, args.names, args.exp_name)
