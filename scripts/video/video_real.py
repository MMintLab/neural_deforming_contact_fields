import argparse
import os
import pdb
import random

import mmint_utils
import numpy as np
import pytorch3d.loss
import torch
import trimesh
from neural_contact_fields.utils import mesh_utils, vedo_utils, utils
from neural_contact_fields.utils.model_utils import load_dataset_from_config
from neural_contact_fields.utils.results_utils import load_gt_results, load_pred_results, load_gt_results_real
from tqdm import trange
from vedo import Plotter, Mesh, Points, LegendBox, Video, settings

settings.showRendererFrame = False
settings.renderer_frame_alpha = 0.0


def vis_mesh_prediction_real(partial_pointcloud: np.ndarray,
                             pred_mesh: trimesh.Trimesh,
                             pred_contact_patch: np.ndarray,
                             gt_contact_patch: np.ndarray,
                             out_fn: str = None
                             ):
    mount_height = 0.036
    tool_height = 0.046

    plt = Plotter(shape=(3, 1))

    # First show the input pointcloud used.
    partial_pc = Points(partial_pointcloud)
    plt.at(0).show(partial_pc)

    # Show predicted mesh, if provided.
    pred_patch_pc = Points(pred_contact_patch, c="red").legend("Predicted")
    pred_geom_mesh = Mesh([pred_mesh.vertices, pred_mesh.faces])
    # plt.at(1).show(pred_geom_mesh, vedo_utils.draw_origin(), "Pred. Mesh", pred_patch_pc)
    plt.at(1).show(pred_geom_mesh)

    pred_patch_pc = Points(pred_contact_patch, c="red", alpha=0.3).legend("Predicted")
    gt_patch_pc = Points(gt_contact_patch, c="blue", alpha=0.4).legend("Ground Truth")
    # leg = LegendBox([pred_patch_pc, gt_patch_pc])
    # plt.at(2).show(pred_patch_pc, gt_patch_pc, leg, vedo_utils.draw_origin(), "Pred. Contact Patch")
    # plt.at(2).show(pred_patch_pc)
    # pred_geom_mesh = Mesh([pred_mesh.vertices, pred_mesh.faces])
    plt.at(2).show(gt_patch_pc, pred_patch_pc)

    plt.camera.SetPosition(0.0798094, 0.113242, 0.170742)
    plt.camera.SetFocalPoint(3.36199e-3, 0.0132810, 0.0688726)
    plt.camera.SetViewUp(0.879073, -0.276793, -0.388093)
    plt.camera.SetDistance(0.161906)
    plt.camera.SetClippingRange(0.102757, 0.231116)

    # if out_fn is not None:
    video = Video(out_fn, backend="ffmpeg", fps=60)

    fps = 60
    sec = 8
    frames = fps * sec
    for i in range(frames):
        angle = (1 / frames) * 2. * np.pi

        partial_pc.rotate_x(angle, rad=True, around=[0.0, 0.0, mount_height + (tool_height / 2.0)])
        pred_geom_mesh.rotate_x(angle, rad=True, around=[0.0, 0.0, mount_height + (tool_height / 2.0)])
        pred_patch_pc.rotate_x(angle, rad=True, around=[0.0, 0.0, mount_height + (tool_height / 2.0)])
        gt_patch_pc.rotate_x(angle, rad=True, around=[0.0, 0.0, mount_height + (tool_height / 2.0)])

        # if out_fn is not None:
        video.add_frame()

    # if out_fn is not None:
    video.close()
    plt.close()


def vis_results(dataset_cfg: str, gen_dir: str, out_dir: str = None, mode: str = "test"):
    mmint_utils.make_dir(out_dir)

    # Load dataset.
    dataset_cfg, dataset = load_dataset_from_config(dataset_cfg, dataset_mode=mode)
    num_trials = len(dataset)

    # Load specific ground truth results needed for evaluation.
    gt_dicts = load_gt_results(dataset, num_trials)

    # Load predicted results.
    pred_dicts = load_pred_results(gen_dir, num_trials)

    # for trial_idx in trange(len(dataset)):
    for trial_idx in [26, 41, 13, 5]:
        # print(trial_idx)
        out_fn = os.path.join(out_dir, "res_%d.mp4" % trial_idx)

        trial_dict = dataset[trial_idx]

        # Load the conditioning pointcloud used.
        pc = trial_dict["partial_pointcloud"]

        vis_mesh_prediction_real(pc,
                                 pred_dicts[trial_idx]["mesh"],
                                 pred_dicts[trial_idx]["contact_patch"],
                                 gt_dicts[trial_idx]["contact_patch"],
                                 out_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize generated results.")
    parser.add_argument("dataset_config", type=str, help="Dataset config.")
    parser.add_argument("gen_dir", type=str, help="Generation directory.")
    parser.add_argument("--out_dir", type=str, default=None, help="Out directory.")
    parser.add_argument("--mode", "-m", type=str, default="test", help="Dataset mode [train, val, test].")
    args = parser.parse_args()

    torch.manual_seed(10)
    np.random.seed(10)
    random.seed(10)

    vis_results(args.dataset_config, args.gen_dir, args.out_dir, args.mode)
