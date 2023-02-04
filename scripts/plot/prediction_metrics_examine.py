import os
import pdb

import mmint_utils
import numpy as np

# base_test_dir = "out/experiments/wrench_v2_tests/partial_pointcloud"
base_test_dir = "out/experiments/terrain_tests_v2/partial_pointcloud"

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

# titles = [
#     "32", "32 (big)", "64", "64 (big)", "16", "16 (big)", "8", "8 (big)", "no wrench", "no wrench (big)",
#     "forward deform", "GR-Net"
# ]
# test_dirs = [
#     "wrench_v1", "wrench_v2", "wrench_v3", "wrench_v4", "wrench_v5", "wrench_v6", "wrench_v7", "wrench_v8",
#     "no_wrench_v1", "no_wrench_v2", "forward_def_v1", "b1_test_final"
# ]

titles = [
    # "8", "16",
    # "32", "64",
    "16",
    "Baseline"
]
test_dirs = [
    # "wrench_v1", "wrench_v2",
    # "wrench_v3", "wrench_v4",
    "wrench_v2",
    "baseline"
]

for title, test_dir in zip(titles, test_dirs):
    metrics_fn = os.path.join(base_test_dir, test_dir, "metrics.pkl.gzip")
    metrics_dict = mmint_utils.load_gzip_pickle(metrics_fn)

    num_examples = len(metrics_dict)

    # Both of these should be defined for all models.
    chamfer_dists = [example["chamfer_distance"] for example in metrics_dict]

    # idx_1 = np.argmin(chamfer_dists[:16])
    # print(idx_1)
    # print(chamfer_dists[idx_1])
    # idx_2 = 16 + np.argmin(chamfer_dists[16:32])
    # print(idx_2)
    # print(chamfer_dists[idx_2])
    # idx_3 = 32 + np.argmin(chamfer_dists[32:])
    # print(idx_3)
    # print(chamfer_dists[idx_3])

    patch_chamfer_dists = [example["patch_chamfer_distance"] for example in metrics_dict]

    print(patch_chamfer_dists)
    print(np.mean())

    # The following are specific to certain models.
    binary_accuracies = [example.get("binary_accuracy", -1.0) for example in metrics_dict]
    precisions = [example.get("precision", -1.0) for example in metrics_dict]
    recalls = [example.get("recall", -1.0) for example in metrics_dict]
    f1 = [example.get("f1", -1.0) for example in metrics_dict]
    ious = [example.get("iou", -1.0) for example in metrics_dict]

    # print(ious[:16])
    # print(np.argmax(ious[:16]))
