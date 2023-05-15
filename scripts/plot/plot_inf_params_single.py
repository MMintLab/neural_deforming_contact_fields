import argparse
import os
from collections import defaultdict

import mmint_utils
import numpy as np


def reject_outliers_from_data(data, m=3):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def perf_to_csv(dirs, out_fn, names=None, reject_outliers=False):
    if names is None:
        names = [os.path.basename(os.path.normpath(d)) for d in dirs]

    out_fn_base = os.path.splitext(out_fn)[0]

    # TODO: Way to generalize this?
    # env_ids = [[0], [1], [2], [0, 1, 2]]
    # env_names = ["box", "curves", "ridges", "all"]
    env_ids = [[0, 1, 2]]
    env_names = ["all"]
    # env_ids = [[0]]
    # env_names = ["real"]
    metrics = ["patch_chamfer_distance", "chamfer_distance", "iou"]

    for env_id_set, env_name in zip(env_ids, env_names):
        out_fn = "%s_%s.csv" % (out_fn_base, env_name)

        keys = dict()
        res_dict = dict()
        for directory, name in zip(dirs, names):
            run_fn = os.path.join(directory, "metrics.pkl.gzip")
            run_dict = mmint_utils.load_gzip_pickle(run_fn)
            run_dict = [d for d in run_dict if d["metadata"]["env_id"] in env_id_set]

            # Collect result on each metric for each example.
            run_res_dict = defaultdict(list)
            for example_idx, example_run_dict in enumerate(run_dict):
                if example_idx in [12, 165]:
                    continue
                example_metrics_dict = example_run_dict["metrics"]

                keys.update(dict.fromkeys(example_metrics_dict.keys()))

                for k, v in example_metrics_dict.items():
                    run_res_dict[k].append(v)

            if reject_outliers:
                for metric in metrics:
                    if metric in run_res_dict:
                        run_res_dict[metric] = reject_outliers_from_data(np.array(run_res_dict[metric]), m=3)

            # Calculate mean/std across examples in run.
            perf_dict = {
                k: {
                    "median": np.median(v),
                    "mean": np.mean(v),
                    "std": np.std(v),
                } for k, v in run_res_dict.items()
            }
            # Calculate rate of patch generation.
            perf_dict["patch_percent"] = len(run_res_dict["patch_chamfer_distance"]) / len(run_dict)
            res_dict[name] = perf_dict

        # Write to CSV.
        csv_str = "dir, name, "
        for key in keys.keys():
            csv_str += "%s, std, median, " % key
        csv_str += "Patch Perc.,"  # Rate of patch generation.
        csv_str += "\n"

        for directory, (name, dir_res_dict) in zip(dirs, res_dict.items()):
            csv_str += "%s, %s, " % (directory, name)
            for key in keys.keys():
                if type(dir_res_dict[key]) is dict:
                    csv_str += "%f, %f, %f, " % \
                               (dir_res_dict[key]["mean"], dir_res_dict[key]["std"], dir_res_dict[key]["median"])

            # Manually add in rate of patch generation.
            csv_str += "%f," % dir_res_dict["patch_percent"]
            csv_str += "\n"

        with open(out_fn, "w") as f:
            f.write(csv_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Consolidate performance metrics.")
    parser.add_argument("dirs", type=str, nargs="+", help="Directories to consolidate.")
    parser.add_argument("out_fn", type=str, help="Output CSV file.")
    parser.add_argument("-n", "--names", type=str, nargs="+", default=None, help="Names for each directory.")
    parser.add_argument("-r", "--reject_outliers", action="store_true", help="Reject outliers.")
    parser.set_defaults(reject_outliers=False)
    args = parser.parse_args()

    perf_to_csv(args.dirs, args.out_fn, args.names, args.reject_outliers)
