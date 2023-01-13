import os
import pdb

import mmint_utils
import numpy as np

base_test_dir = "out/experiments/wrench_tests"

# test_dirs = ["wrench_v4", "wrench_v1", "wrench_v2"]
# titles = ["l=3", "l=6", "l=12"]

test_dirs = ["wrench_v7", "wrench_v1", "wrench_v3"]
titles = ["lr=1e-3", "lr=1e-4", "lr=1e-5"]

# test_dirs = ["wrench_v5", "wrench_v1", "wrench_v6"]
# titles = ["l2=0.1", "l2=1e-3", "l2=1e-5"]

for title, test_dir in zip(titles, test_dirs):
    metrics_fn = os.path.join(base_test_dir, test_dir, "metrics.pkl.gzip")
    metrics_dict = mmint_utils.load_gzip_pickle(metrics_fn)

    num_examples = len(metrics_dict)

    chamfer_dists = [example["chamfer_distance"][0].item() for example in metrics_dict]
    binary_accuracies = [example["binary_accuracy"].item() for example in metrics_dict]
    precisions = [float(example["pr"]["precision"]) for example in metrics_dict]
    recalls = [float(example["pr"]["recall"]) for example in metrics_dict]

    print("Title: %s. Binary Accuracy: %f (%f). Chamfer Dist: %f (%f). Precision: %f (%f). Recall: %f (%f)"
          % (title, np.mean(binary_accuracies), np.std(binary_accuracies),
             np.mean(chamfer_dists), np.std(chamfer_dists),
             np.mean(precisions), np.std(precisions),
             np.mean(recalls), np.std(recalls)
             ))
