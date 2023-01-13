import os
import pdb

import mmint_utils

base_test_dir = "out/experiments/wrench_tests"

# test_dirs = ["wrench_v4", "wrench_v1", "wrench_v2"]
# titles = ["l=3", "l=6", "l=12"]

# test_dirs = ["wrench_v7", "wrench_v1", "wrench_v3"]
# titles = ["lr=1e-3", "lr=1e-4", "lr=1e-5"]

test_dirs = ["wrench_v5", "wrench_v1", "wrench_v6"]
titles = ["l2=0.1", "l2=1e-3", "l2=1e-5"]

for title, test_dir in zip(titles, test_dirs):
    metrics_fn = os.path.join(base_test_dir, test_dir, "metrics.pkl.gzip")
    metrics_dict = mmint_utils.load_gzip_pickle(metrics_fn)

    num_examples = len(metrics_dict)

    chamfer_dist = 0.0
    binary_accuracy = 0.0
    precision = 0.0
    recall = 0.0

    for example_idx in range(num_examples):
        chamfer_dist += metrics_dict[example_idx]["chamfer_distance"][0].item()
        binary_accuracy += metrics_dict[example_idx]["binary_accuracy"].item()
        precision += metrics_dict[example_idx]["pr"]["precision"].item()
        recall += metrics_dict[example_idx]["pr"]["recall"].item()

    chamfer_dist /= num_examples
    binary_accuracy /= num_examples
    precision /= num_examples
    recall /= num_examples

    print("Title: %s. Binary Accuracy: %f. Chamfer Dist: %f. Precision: %f. Recall: %f"
          % (title, binary_accuracy, chamfer_dist, precision, recall))
