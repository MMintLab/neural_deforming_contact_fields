import os

import mmint_utils
import numpy as np
import torch
import trimesh
from neural_contact_fields.utils import utils


def write_nominal_results(out_dir, nominal_mesh, idx, misc: dict = None):
    if nominal_mesh is not None:
        mesh_fn = os.path.join(out_dir, "nominal_mesh_%d.obj" % idx)
        nominal_mesh.export(mesh_fn)

    if misc is not None:
        misc_fn = os.path.join(out_dir, "misc_%d.pkl.gzip" % idx)
        mmint_utils.save_gzip_pickle(misc, misc_fn)


def write_results(out_dir, gen_dict: dict, idx):
    mesh = gen_dict["mesh"]
    if mesh is not None:
        mesh_fn = os.path.join(out_dir, "mesh_%d.obj" % idx)
        mesh.export(mesh_fn)

    pointcloud = gen_dict["pointcloud"]
    if pointcloud is not None:
        pc_fn = os.path.join(out_dir, "pointcloud_%d.ply" % idx)
        utils.save_pointcloud(pointcloud, pc_fn)

    contact_patch = gen_dict["contact_patch"]
    if contact_patch is not None:
        cp_fn = os.path.join(out_dir, "contact_patch_%d.ply" % idx)
        utils.save_pointcloud(contact_patch, cp_fn)

    contact_labels = gen_dict["contact_labels"]
    if contact_labels is not None:
        cl_fn = os.path.join(out_dir, "contact_labels_%d.pkl.gzip" % idx)
        mmint_utils.save_gzip_pickle(contact_labels, cl_fn)

    if "iou_labels" in gen_dict:
        iou_labels = gen_dict["iou_labels"]
        if iou_labels is not None:
            iou_fn = os.path.join(out_dir, "iou_labels_%d.pkl.gzip" % idx)
            mmint_utils.save_gzip_pickle(iou_labels, iou_fn)

    metadata = gen_dict["metadata"]
    if metadata is not None:
        # Make sure mesh not in metadata.
        if "mesh" in metadata:
            del metadata["mesh"]
        metadata_fn = os.path.join(out_dir, "misc_%d.pkl.gzip" % idx)
        mmint_utils.save_gzip_pickle(metadata, metadata_fn)


def load_pred_results(out_dir, n, device=None):
    gen_dicts = []

    for idx in range(n):
        gen_dict = dict()

        mesh_fn = os.path.join(out_dir, "mesh_%d.obj" % idx)
        gen_dict["mesh"] = trimesh.load(mesh_fn) if os.path.exists(mesh_fn) else None

        pc_fn = os.path.join(out_dir, "pointcloud_%d.ply" % idx)
        gen_dict["pointcloud"] = torch.from_numpy(utils.load_pointcloud(pc_fn)).to(device) if os.path.exists(
            pc_fn) else None

        cp_fn = os.path.join(out_dir, "contact_patch_%d.ply" % idx)
        gen_dict["contact_patch"] = torch.from_numpy(utils.load_pointcloud(cp_fn)).to(device) if os.path.exists(
            cp_fn) else None

        cl_fn = os.path.join(out_dir, "contact_labels_%d.pkl.gzip" % idx)
        gen_dict["contact_labels"] = mmint_utils.load_gzip_pickle(cl_fn) if os.path.exists(cl_fn) else None

        iou_fn = os.path.join(out_dir, "iou_labels_%d.pkl.gzip" % idx)
        gen_dict["iou_labels"] = mmint_utils.load_gzip_pickle(iou_fn) if os.path.exists(iou_fn) else None

        misc_fn = os.path.join(out_dir, "misc_%d.pkl.gzip" % idx)
        gen_dict["metadata"] = mmint_utils.load_gzip_pickle(misc_fn) if os.path.exists(misc_fn) else None

        gen_dicts.append(gen_dict)
    return gen_dicts


def load_gt_results(dataset, n, device=None):
    gt_dicts = []

    # Load ground truth meshes and surface contact labels.
    for idx in range(n):
        dataset_dict = dataset[idx]
        gt_dict = dict()

        gt_dict["mesh"] = dataset.get_example_mesh(idx)
        gt_dict["pointcloud"] = torch.from_numpy(dataset_dict["surface_points"]).to(device)
        gt_dict["contact_patch"] = torch.from_numpy(dataset_dict["contact_patch"]).to(device)
        gt_dict["contact_labels"] = torch.from_numpy(dataset_dict["surface_in_contact"]).to(device).int()
        gt_dict["iou_labels"] = torch.from_numpy(dataset_dict["occ_tgt"]).to(device).int()
        gt_dict["points_iou"] = torch.from_numpy(dataset_dict["points_iou"]).to(device)

        gt_dicts.append(gt_dict)

    return gt_dicts


def load_gt_results_real(dataset, dataset_dir, n, device=None):
    contact_patches = []

    # Load ground truth meshes and surface contact labels.
    for idx in range(n):
        dataset_dict = dataset[idx]

        # Load contact patch.
        contact_patch = torch.from_numpy(dataset_dict["contact_patch"]).to(device).float()
        contact_patches.append(contact_patch)

    return contact_patches


def metrics_to_statistics(metrics_dicts):
    keys = metrics_dicts[0].keys()

    statistics = dict()
    for key in keys:
        statistics[f"{key}_mean"] = np.mean([example[key] for example in metrics_dicts if example[key] is not None])
        statistics[f"{key}_std"] = np.std([example[key] for example in metrics_dicts if example[key] is not None])

    return statistics


def print_results(metrics_dict, title):
    chamfer_dists = [example["chamfer_distance"][0].item() for example in metrics_dict]
    binary_accuracies = [example["binary_accuracy"].item() for example in metrics_dict]
    precisions = [float(example["pr"]["precision"]) for example in metrics_dict]
    recalls = [float(example["pr"]["recall"]) for example in metrics_dict]
    ious = [float(example["iou"]) for example in metrics_dict]

    print(
        "Title: %s. Binary Accuracy: %f (%f). Chamfer Dist: %f (%f). IoU: %f (%f). Precision: %f (%f). Recall: %f (%f)."
        % (title, np.mean(binary_accuracies), np.std(binary_accuracies),
           np.mean(chamfer_dists), np.std(chamfer_dists),
           np.mean(ious), np.std(ious),
           np.mean(precisions), np.std(precisions),
           np.mean(recalls), np.std(recalls)
           ))
