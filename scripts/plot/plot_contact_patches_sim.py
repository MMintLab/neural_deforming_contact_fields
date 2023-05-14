import os

from neural_contact_fields.utils.model_utils import load_dataset_from_config
from neural_contact_fields.utils.results_utils import load_gt_results, load_pred_results
from tqdm import trange
from vedo import Plotter, Points

dataset_cfg = "cfg/primitives/camera_ready/dataset/test_v1.yaml"
mode = "test"
base_test_dir = "out/experiments/camera_ready/sim/test_v1/"

titles = [
    "Ours",
    "Baseline"
]
test_dirs = [
    "model_v3",
    "baseline"
]

# Load dataset.
dataset_cfg, dataset = load_dataset_from_config(dataset_cfg, dataset_mode=mode)
num_trials = len(dataset)

# Load specific ground truth results needed for evaluation.
gt_dicts = load_gt_results(dataset, num_trials)
gt_contact_patches = [gt_dict["contact_patch"] for gt_dict in gt_dicts]

pred_contact_patches_all = []
for title, test_dir in zip(titles, test_dirs):
    gen_dir = os.path.join(base_test_dir, test_dir)
    # Load predicted results.
    gen_dicts = load_pred_results(gen_dir, num_trials)
    pred_contact_patches = [gen_dict["contact_patch"] for gen_dict in gen_dicts]

    pred_contact_patches_all.append(pred_contact_patches)

# Build plots.
# for index in trange(num_trials):
for index in [80, 16, 107, 123, 207, 228]:
    print(index)
    trial_dict = dataset[index]

    plt = Plotter(shape=(1, 3))

    for method_idx in range(len(titles)):
        pred_patch = pred_contact_patches_all[method_idx][index]
        plt.at(method_idx).show(Points(pred_patch, c="red"))

    plt.at(len(titles)).show(Points(gt_contact_patches[index], c="blue"))
    plt.interactive().close()
