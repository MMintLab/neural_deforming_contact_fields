import os

import trimesh

import mmint_utils
from neural_contact_fields.utils import utils
from neural_contact_fields.utils.model_utils import load_dataset_from_config
from neural_contact_fields.utils.results_utils import load_gt_results, load_pred_results
from tqdm import trange
from vedo import Plotter, Points, Mesh, Video

from scripts.video.vedo_animator import VedoOrbiter

dataset_cfg = "cfg/primitives/camera_ready/dataset/test_v1.yaml"
mode = "test"
base_test_dir = "out/experiments/camera_ready/sim/test_v1/"
terrain_dir = "/home/markvdm/Documents/IsaacGym/ncf_envs/out/primitives/camera_ready/combine_test/"
out_dir = "/home/markvdm/Pictures/RSS_CR/Video/Sim/"

titles = [
    "Ours",
    "Baseline"
]
test_dirs = [
    "model_v3",
    "baseline"
]

no_animate = False

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

for trial_idx in [80, 16, 107, 123, 207, 228]:
    # print(trial_idx)
    data_dict = mmint_utils.load_gzip_pickle(os.path.join(terrain_dir, "config_%d.pkl.gzip" % trial_idx))

    # Get wrist pose.
    wrist_pose = data_dict["wrist_pose"]
    w_T_wrist_pose = utils.pose_to_matrix(wrist_pose, axes="rxyz")
    trial_dict = dataset[trial_idx]

    terrain_obj_fn = os.path.join(terrain_dir, "terrain_%d.obj" % trial_idx)
    terrain_mesh = trimesh.load_mesh(terrain_obj_fn)
    terrain_mesh_vis = Mesh([terrain_mesh.vertices, terrain_mesh.faces], c="gray", alpha=0.2)

    plt = Plotter(shape=(1, 4))
    plt.at(0).show(Points(utils.transform_pointcloud(trial_dict["partial_pointcloud"], w_T_wrist_pose), c="blue"),
                   terrain_mesh_vis)

    vis_items = []
    for method_idx in range(len(titles)):
        pred_mesh = pred_meshes_all[method_idx][trial_idx]
        pred_mesh.apply_transform(w_T_wrist_pose)
        pred_mesh_vis = Mesh([pred_mesh.vertices, pred_mesh.faces])

        pred_patch = pred_contact_patches_all[method_idx][trial_idx]
        pred_patch = utils.transform_pointcloud(pred_patch, w_T_wrist_pose)
        pred_patch_vis = Points(pred_patch, c="red")

        vis_items_idx = [pred_mesh_vis, pred_patch_vis, terrain_mesh_vis]

        plt.at(method_idx + 1).show(*vis_items_idx)
        vis_items.append(vis_items_idx)

    gt_mesh = gt_meshes[trial_idx]
    gt_mesh.apply_transform(w_T_wrist_pose)
    gt_mesh_vis = Mesh([gt_mesh.vertices, gt_mesh.faces])
    plt.at(len(titles) + 1).show(gt_mesh_vis,
                                 Points(utils.transform_pointcloud(gt_contact_patches[trial_idx], w_T_wrist_pose),
                                        c="red"),
                                 terrain_mesh_vis)

    if no_animate:
        plt.interactive().close()
    else:
        orbiter = VedoOrbiter(plt, None, period=8, dist=0.2, pitch=0.4, target=gt_mesh.centroid)

        fps = int(1.0 / orbiter.update_period)
        num_per_spin = int(fps * orbiter.period)

        video = Video(os.path.join(out_dir, "vis_%d.mp4" % trial_idx), backend="ffmpeg", fps=fps)

        for spin_idx in range(num_per_spin):
            orbiter.update_transform(orbiter.generate_transform(orbiter.update_period * spin_idx))

            video.add_frame()

        for spin_idx in range(num_per_spin):
            orbiter.update_transform(orbiter.generate_transform(orbiter.update_period * spin_idx))
            for vis_items_idx in vis_items:
                vis_items_idx[0].alpha(max(0.2, 1.0 - (spin_idx / (num_per_spin / 10.0)) * 0.8))
            gt_mesh_vis.alpha(max(0.2, 1.0 - (spin_idx / (num_per_spin / 10.0)) * 0.8))

            # sleep(0.01)
            video.add_frame()

        for spin_idx in range(num_per_spin):
            orbiter.update_transform(orbiter.generate_transform(orbiter.update_period * spin_idx))

            for vis_items_idx in vis_items:
                vis_items_idx[0].alpha(min(1.0, 0.2 + (spin_idx / (num_per_spin / 10.0)) * 0.8))
            gt_mesh_vis.alpha(min(1.0, 0.2 + (spin_idx / (num_per_spin / 10.0)) * 0.8))

            video.add_frame()

        video.close()
        plt.close()
