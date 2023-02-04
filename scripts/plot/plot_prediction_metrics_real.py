import os
import pdb

import mmint_utils
import numpy as np

# base_test_dir = "out/experiments/wrench_v2_tests/partial_pointcloud"
# base_test_dir = "out/experiments/real_world_test/test_loss_weights_sim"
base_test_dir = "out/experiments/real_terrain_test/partial_pointcloud"

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

# titles = [
#     "W=0.01", "W=0.1", "W=1.0", "W=10.0", "W=100.0", "W=1000.0"
# ]
# test_dirs = [
#     "exp_v1", "exp_v2", "exp_v3", "exp_v4", "exp_v5", "exp_v6"
# ]

titles = [
    "Ours (l=16)", "Baseline"
]
test_dirs = [
    "wrench_v2", "baseline"
]

out_fn = os.path.join(base_test_dir, "out.csv")
csv_str = ""

csv_str += "Config, Run, Patch CD, Std.\n"

for title, test_dir in zip(titles, test_dirs):
    metrics_fn = os.path.join(base_test_dir, test_dir, "metrics.pkl.gzip")
    metrics_dict = mmint_utils.load_gzip_pickle(metrics_fn)

    num_examples = len(metrics_dict)

    # Both of these should be defined for all models.
    patch_chamfer_dists = [example["patch_chamfer_distance"] for example in metrics_dict]
    print(patch_chamfer_dists)

    csv_str += "%s, %s, %f, %f,\n" % \
               (test_dir, title,
                np.mean(patch_chamfer_dists), np.std(patch_chamfer_dists),
                )

with open(out_fn, "w") as f:
    f.write(csv_str)
