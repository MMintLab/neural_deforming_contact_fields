import argparse
import os
from collections import defaultdict

import mmint_utils
import numpy as np


def perf_to_csv(dirs, out_fn, names=None):
    if names is None:
        names = [os.path.basename(os.path.normpath(d)) for d in dirs]

    keys = None
    res_dict = {}
    for directory, name in zip(dirs, names):
        run_dirs = [f for f in os.listdir(directory) if "run_" in f]

        res_dicts = []
        for run_dir in run_dirs:
            run_out_dir = os.path.join(directory, run_dir)
            run_fn = os.path.join(run_out_dir, "metrics.pkl.gzip")
            run_dict = mmint_utils.load_gzip_pickle(run_fn)

            # Collect result on each metric for each example.
            run_res_dict = defaultdict(list)
            for example_run_dict in run_dict:
                example_metrics_dict = example_run_dict["metrics"]

                if keys is None:
                    keys = example_metrics_dict.keys()

                for k, v in example_metrics_dict.items():
                    run_res_dict[k].append(v)

            # Calculate mean across examples in run.
            run_res_dict = {k: np.mean(v) for k, v in run_res_dict.items()}
            res_dicts.append(run_res_dict)

        # Calculate mean across runs.
        dir_res_dict = {
            k: {
                "mean": np.mean([d[k] for d in res_dicts]),
                "std": np.std([d[k] for d in res_dicts]),
            }
            for k in keys
        }
        res_dict[name] = {
            "metrics": dir_res_dict,
        }

    # Write to CSV.
    csv_str = "dir, name, "
    for key in keys:
        csv_str += "%s, std, " % key
    csv_str += "\n"

    for directory, (name, dir_res_dict) in zip(dirs, res_dict.items()):
        csv_str += "%s, %s, " % (directory, name)
        for key in keys:
            csv_str += "%f, %f, " % (dir_res_dict[key]["mean"], dir_res_dict[key]["std"])
        csv_str += "\n"

    with open(out_fn, "w") as f:
        f.write(csv_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Consolidate performance metrics.")
    parser.add_argument("dirs", type=str, nargs="+", help="Directories to consolidate.")
    parser.add_argument("out_fn", type=str, help="Output CSV file.")
    parser.add_argument("-n", "--names", type=str, nargs="+", default=None, help="Names for each directory.")
    args = parser.parse_args()

    perf_to_csv(args.dirs, args.out_fn, args.names)
