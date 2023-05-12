import argparse
import os
from collections import defaultdict

from matplotlib import pyplot as plt

import mmint_utils
import numpy as np


def perf_to_csv(dirs, names=None):
    if names is None:
        names = [os.path.basename(os.path.normpath(d)) for d in dirs]

    # TODO: Way to generalize this?
    env_ids = [[0, 1, 2]]
    env_names = ["all"]
    metrics = ["patch_chamfer_distance", "chamfer_distance", "iou"]

    for env_id_set, env_name in zip(env_ids, env_names):
        res_dict = dict()
        for directory, name in zip(dirs, names):
            run_fn = os.path.join(directory, "metrics.pkl.gzip")
            run_dict = mmint_utils.load_gzip_pickle(run_fn)
            run_dict = [d for d in run_dict if d["metadata"]["env_id"] in env_id_set]

            # Collect result on each metric for each example.
            run_res_dict = defaultdict(list)
            for example_run_dict in run_dict:
                example_metrics_dict = example_run_dict["metrics"]

                for metric in metrics:
                    if metric in example_metrics_dict:
                        run_res_dict[metric].append(example_metrics_dict[metric])

            res_dict[name] = run_res_dict

        # Build boxplot data.
        boxplot_data = []
        for name in names:
            boxplot_data.append(res_dict[name]["patch_chamfer_distance"])
        plt.boxplot(boxplot_data, vert="True")
        plt.xticks(np.arange(1, len(names) + 1), names)
        plt.ylabel("Patch Chamfer Distance")
        plt.xlabel("Run")
        plt.title("Patch Chamfer Distance for %s" % env_name)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Consolidate performance metrics.")
    parser.add_argument("dirs", type=str, nargs="+", help="Directories to consolidate.")
    parser.add_argument("-n", "--names", type=str, nargs="+", default=None, help="Names for each directory.")
    args = parser.parse_args()

    perf_to_csv(args.dirs, args.names)
