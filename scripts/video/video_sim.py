import argparse
import os

import mmint_utils
import numpy as np
import trimesh
from neural_contact_fields.utils import mesh_utils, vedo_utils, utils
from neural_contact_fields.utils.model_utils import load_dataset_from_config
from neural_contact_fields.utils.results_utils import load_gt_results, load_pred_results
from tqdm import trange
from vedo import Plotter, Mesh, Points, LegendBox, Video, settings

# settings.showRendererFrame = False
# settings.renderer_frame_alpha = 0.0

mount_height = 0.036
tool_height = 0.046
dataset_cfg = "cfg/primitives/dataset/test_rebuttal.yaml"
# dataset_cfg = "cfg/dataset/sim_terrain_test_v2.yaml"
mode = "test"
base_test_dir = "out/experiments/rebuttal/inference_timing"
video_dir = "/home/markvdm/Pictures/RSS_Rebuttal/video/sim_results/"

# base_test_dir = "out/experiments/terrain_tests_v3/partial_pointcloud"
# video_dir = "/home/markvdm/Pictures/RSS_2023/video/sim_results/"
mmint_utils.make_dir(video_dir)

titles = [
    "Ours",
]
test_dirs = [
    "finetuned"
#    "wrench_v2",
]

# Load dataset.
dataset_cfg, dataset = load_dataset_from_config(dataset_cfg, dataset_mode=mode)
num_trials = len(dataset)

# Load specific ground truth results needed for evaluation.
gt_meshes, gt_pointclouds, gt_contact_patches, gt_contact_labels, points_iou, gt_occ_iou = load_gt_results(
    dataset, dataset_cfg["data"][mode]["dataset_dir"], num_trials
)

pred_meshes_all = []
pred_patches_all = []
for title, test_dir in zip(titles, test_dirs):
    gen_dir = os.path.join(base_test_dir, test_dir)
    # Load predicted results.
    pred_meshes, pred_pointclouds, pred_contact_patches, pred_contact_labels, _, _ = load_pred_results(gen_dir, num_trials)

    pred_meshes_all.append(pred_meshes)
    pred_patches_all.append(pred_contact_patches)

# Build plots.
for index in range(38, num_trials):
    trial_dict = dataset[index]
    pc = trial_dict["partial_pointcloud"]

    vedo_elements = []

    plt = Plotter(shape=(2, 4))
    partial_pc = Points(pc, c="black")
    plt.at(0).show(partial_pc)
    vedo_elements.append(partial_pc)

    # Plot geometries.
    # print(len(titles), len(pred_meshes_all), index, len(pred_meshes_all[0])
    for method_idx in range(len(titles)):
        pred_mesh_ = pred_meshes_all[method_idx][index]
        pred_mesh = Mesh([pred_mesh_.vertices, pred_mesh_.faces])
        vedo_elements.append(pred_mesh)
        plt.at(1 + method_idx).show(pred_mesh)

    gt_mesh_ = gt_meshes[index]
    gt_mesh = Mesh([gt_mesh_.vertices, gt_mesh_.faces], c="grey")
    vedo_elements.append(gt_mesh)
    plt.at(1 + len(titles)).show(gt_mesh)

    # Plot contact patches.
    for method_idx in range(len(titles)):
        pred_patch_ = pred_patches_all[method_idx][index]
        pred_patch = Points(pred_patch_, c="red")
        vedo_elements.append(pred_patch)
        plt.at(5 + method_idx).show(pred_patch)

    gt_patch = Points(gt_contact_patches[index], c="blue")
    vedo_elements.append(gt_patch)
    plt.at(5 + len(titles)).show(gt_patch)

    plt.camera.SetPosition(0.0798094, 0.113242, 0.170742)
    plt.camera.SetFocalPoint(3.36199e-3, 0.0132810, 0.0688726)
    plt.camera.SetViewUp(0.879073, -0.276793, -0.388093)
    plt.camera.SetDistance(0.161906)
    plt.camera.SetClippingRange(0.102757, 0.231116)

    # if out_fn is not None:
    video = Video(os.path.join(video_dir, "res_%d.mp4" % index), backend="ffmpeg", fps=60)

    fps = 60
    sec = 8
    frames = fps * sec
    for i in range(frames):
        angle = (1 / frames) * 2. * np.pi

        for vedo_el in vedo_elements:
            vedo_el.rotate_x(angle, rad=True, around=[0.0, 0.0, mount_height + (tool_height / 2.0)])

        # if out_fn is not None:
        video.add_frame()

    # if out_fn is not None:
    video.close()
    plt.close()
