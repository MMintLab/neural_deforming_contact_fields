import argparse
import os

from neural_contact_fields.utils.model_utils import load_dataset_from_config
from neural_contact_fields.utils.results_utils import load_gt_results, load_pred_results
from tqdm import trange
from vedo import Plotter, Mesh, Points

parser = argparse.ArgumentParser(description="Plot predicted and ground truth geometries for sim results.")
parser.add_argument("--offset", "-o", type=int, default=0, help="Offset to start plotting from.")
args = parser.parse_args()
offset = args.offset

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
gt_meshes = [gt_dict["mesh"] for gt_dict in gt_dicts]
gt_contact_patches = [gt_dict["contact_patch"] for gt_dict in gt_dicts]

pred_meshes_all = []
pred_contact_patches_all = []
for title, test_dir in zip(titles, test_dirs):
    gen_dir = os.path.join(base_test_dir, test_dir)
    # Load predicted results.
    gen_dicts = load_pred_results(gen_dir, num_trials)
    pred_meshes = [gen_dict["mesh"] for gen_dict in gen_dicts]
    pred_contact_patches = [gen_dict["contact_patch"] for gen_dict in gen_dicts]

    pred_meshes_all.append(pred_meshes)
    pred_contact_patches_all.append(pred_contact_patches)

# Build plots.
for index in trange(offset, num_trials):
    trial_dict = dataset[index]
    pc = trial_dict["partial_pointcloud"]

    plt = Plotter(shape=(1, 4))
    plt.at(0).show(Points(pc, c="blue", r=2))

    for method_idx in range(len(titles)):
        pred_mesh = pred_meshes_all[method_idx][index]
        pred_patch = pred_contact_patches_all[method_idx][index]
        plt.at(1 + method_idx).show(Mesh([pred_mesh.vertices, pred_mesh.faces]))

    plt.at(1 + len(titles)).show(gt_meshes[index])
    plt.interactive().close()
