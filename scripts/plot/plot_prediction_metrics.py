import os
import pdb

import mmint_utils
import numpy as np

base_test_dir = "out/experiments/wrench_v2_tests/partial_pointcloud"

# test_dirs = ["wrench_v4", "wrench_v1", "wrench_v2"]
# titles = ["l=3", "l=6", "l=12"]

# test_dirs = ["wrench_v7", "wrench_v1", "wrench_v3"]
# titles = ["lr=1e-3", "lr=1e-4", "lr=1e-5"]
#
# test_dirs = ["wrench_v5", "wrench_v1", "wrench_v6"]
# titles = ["l2=0.1", "l2=1e-3", "l2=1e-5"]

# test_dirs = ["wrench_v8", "wrench_v9", "wrench_v10"]
# titles = ["latent=16", "latent=32", "latent=64"]

# titles = [
#     "l=3", "l=6", "l=12", "lr=1e-3", "lr=1e-4", "lr=1e-5", "l2=0.1", "l2=1e-3", "l2=1e-5", "latent=16", "latent=32",
#     "latent=64", "no wrench", "mlp"
# ]
# test_dirs = [
#     "wrench_v4", "wrench_v1", "wrench_v2", "wrench_v7", "wrench_v1", "wrench_v3", "wrench_v5", "wrench_v1", "wrench_v6",
#     "wrench_v8", "wrench_v9", "wrench_v10", "no_wrench_v1", "mlp_v1"
# ]

titles = [
    "32", "32 (big)", "64", "64 (big)", "no wrench", "no wrench (big)"
]
test_dirs = [
    "wrench_v1", "wrench_v2", "wrench_v3", "wrench_v4", "no_wrench_v1", "no_wrench_v2"
]

out_fn = "out/experiments/wrench_v2_tests/partial_pointcloud/out.csv"
csv_str = ""

for title, test_dir in zip(titles, test_dirs):
    metrics_fn = os.path.join(base_test_dir, test_dir, "metrics.pkl.gzip")
    metrics_dict = mmint_utils.load_gzip_pickle(metrics_fn)

    num_examples = len(metrics_dict)

    chamfer_dists = [example["chamfer_distance"] for example in metrics_dict]
    binary_accuracies = [example["binary_accuracy"] for example in metrics_dict]
    precisions = [example["precision"] for example in metrics_dict]
    recalls = [example["recall"] for example in metrics_dict]
    ious = [example["iou"] for example in metrics_dict]

    # print(
    #     "Title: %s. Binary Accuracy: %f (%f). Chamfer Dist: %f (%f). IoU: %f (%f). Precision: %f (%f). Recall: %f (%f)."
    #     % (title, np.mean(binary_accuracies), np.std(binary_accuracies),
    #        np.mean(chamfer_dists), np.std(chamfer_dists),
    #        np.mean(ious), np.std(ious),
    #        np.mean(precisions), np.std(precisions),
    #        np.mean(recalls), np.std(recalls)
    #        ))
    csv_str += "%s, %f (%f), %f (%f), %f (%f), %f (%f), %f (%f)\n" % \
               (title, np.mean(binary_accuracies), np.std(binary_accuracies),
                np.mean(chamfer_dists), np.std(chamfer_dists),
                np.mean(ious), np.std(ious),
                np.mean(precisions), np.std(precisions),
                np.mean(recalls), np.std(recalls)
                )

with open(out_fn, "w") as f:
    f.write(csv_str)
