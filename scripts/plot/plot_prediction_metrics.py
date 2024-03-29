import os
import pdb

import mmint_utils
import numpy as np

# base_test_dir = "out/experiments/wrench_v2_tests/partial_pointcloud"
# base_test_dir = "out/experiments/terrain_tests_v3/partial_pointcloud"
base_test_dir = "out/models/primitives/camera_ready/model_v1/out/generate/"

out_name = "val_v2_%s.csv"

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
    # "Ours",
    # "Baseline",
    # "No Wrench",
    # "Forward Deformation"
    "Random", "Random (High-Res)"
]
test_dirs = [
    # "wrench_v1", "wrench_v2",
    # "wrench_v3", "wrench_v4",
    # "wrench_v2",
    # "baseline",
    # "no_wrench_v1",
    # "forward_def_v1",
    "val_v2_random", "val_v2_random_high_res"
]

env_ids = [[0], [1], [2], [0, 1, 2]]
env_names = ["box", "curves", "ridges", "all"]

for env_id_set, env_name in zip(env_ids, env_names):
    out_fn = os.path.join(base_test_dir, out_name % env_name)
    csv_str = ""

    csv_str += "Config, Run, Geometry CD, Std., IoU, Std., Patch CD, Std., Patch Perc., Binary Acc., Std., Precision, Std., Recall," \
               " Std., F1, Std.\n"

    for title, test_dir in zip(titles, test_dirs):
        metrics_fn = os.path.join(base_test_dir, test_dir, "metrics.pkl.gzip")
        metrics_dict = mmint_utils.load_gzip_pickle(metrics_fn)

        metrics_dict = [example for example in metrics_dict if example["env_class"] in env_id_set]

        num_examples = len(metrics_dict)

        # Both of these should be defined for all models.
        chamfer_dists = [example["chamfer_distance"] for example in metrics_dict]
        patch_chamfer_dists = [example["patch_chamfer_distance"] for example in metrics_dict if
                               "patch_chamfer_distance" in example.keys()]
        num_patches = len(patch_chamfer_dists)
        percent_generated = num_patches / num_examples

        # print(title)
        # print("Discrete: %f (%f)" % (np.mean(patch_chamfer_dists[:16]), np.std(patch_chamfer_dists[:16])))
        # print("Wave: %f (%f)" % (np.mean(patch_chamfer_dists[16:32]), np.std(patch_chamfer_dists[16:32])))
        # print("Flat: %f (%f)" % (np.mean(patch_chamfer_dists[32:]), np.std(patch_chamfer_dists[32:])))

        # The following are specific to certain models.
        binary_accuracies = [example.get("binary_accuracy", -1.0) for example in metrics_dict]
        precisions = [example.get("precision", -1.0) for example in metrics_dict]
        recalls = [example.get("recall", -1.0) for example in metrics_dict]
        f1 = [example.get("f1", -1.0) for example in metrics_dict]
        ious = [example.get("iou", -1.0) for example in metrics_dict]

        csv_str += "%s, %s, %f, %f, %f,%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f,\n" % \
                   (test_dir, title,
                    np.mean(chamfer_dists), np.std(chamfer_dists),
                    np.mean(ious), np.std(ious),
                    np.mean(patch_chamfer_dists), np.std(patch_chamfer_dists), percent_generated,
                    np.mean(binary_accuracies), np.std(binary_accuracies),
                    np.mean(precisions), np.std(precisions),
                    np.mean(recalls), np.std(recalls),
                    np.mean(f1), np.std(f1),
                    )

    with open(out_fn, "w") as f:
        f.write(csv_str)
