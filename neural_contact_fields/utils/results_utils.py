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


def write_results(out_dir, mesh, pointcloud, contact_patch, contact_labels, iou_labels, idx, misc: dict = None):
    if mesh is not None:
        mesh_fn = os.path.join(out_dir, "mesh_%d.obj" % idx)
        mesh.export(mesh_fn)

    if pointcloud is not None:
        pc_fn = os.path.join(out_dir, "pointcloud_%d.ply" % idx)
        utils.save_pointcloud(pointcloud, pc_fn)

    if contact_patch is not None:
        cp_fn = os.path.join(out_dir, "contact_patch_%d.ply" % idx)
        utils.save_pointcloud(contact_patch, cp_fn)

    if contact_labels is not None:
        cl_fn = os.path.join(out_dir, "contact_labels_%d.pkl.gzip" % idx)
        mmint_utils.save_gzip_pickle(contact_labels, cl_fn)

    if iou_labels is not None:
        iou_fn = os.path.join(out_dir, "iou_labels_%d.pkl.gzip" % idx)
        mmint_utils.save_gzip_pickle(iou_labels, iou_fn)

    if misc is not None:
        misc_fn = os.path.join(out_dir, "misc_%d.pkl.gzip" % idx)
        mmint_utils.save_gzip_pickle(misc, misc_fn)


def load_pred_results(out_dir, n, device=None):
    meshes = []
    pointclouds = []
    contact_patches = []
    contact_labels = []
    iou_labels = []
    miscs = []

    for idx in range(n):
        mesh_fn = os.path.join(out_dir, "mesh_%d.obj" % idx)
        if os.path.exists(mesh_fn):
            meshes.append(trimesh.load(mesh_fn))
        else:
            meshes.append(None)

        pc_fn = os.path.join(out_dir, "pointcloud_%d.ply" % idx)
        if os.path.exists(pc_fn):
            pointclouds.append(torch.from_numpy(utils.load_pointcloud(pc_fn)).to(device))
        else:
            pointclouds.append(None)

        cp_fn = os.path.join(out_dir, "contact_patch_%d.ply" % idx)
        if os.path.exists(cp_fn):
            contact_patches.append(torch.from_numpy(utils.load_pointcloud(cp_fn)).to(device))
        else:
            contact_patches.append(None)

        cl_fn = os.path.join(out_dir, "contact_labels_%d.pkl.gzip" % idx)
        if os.path.exists(cl_fn):
            contact_labels_ = mmint_utils.load_gzip_pickle(cl_fn)
            contact_labels.append(contact_labels_)
        else:
            contact_labels.append(None)

        iou_fn = os.path.join(out_dir, "iou_labels_%d.pkl.gzip" % idx)
        if os.path.exists(iou_fn):
            iou_labels_ = mmint_utils.load_gzip_pickle(iou_fn)
            iou_labels.append(iou_labels_)
        else:
            iou_labels.append(None)

        misc_fn = os.path.join(out_dir, "misc_%d.pkl.gzip" % idx)
        if os.path.exists(misc_fn):
            misc = mmint_utils.load_gzip_pickle(misc_fn)
            miscs.append(misc)
        else:
            miscs.append(None)

    return meshes, pointclouds, contact_patches, contact_labels, iou_labels, miscs


def load_gt_results(dataset, dataset_dir, n, device=None):
    meshes = []
    pointclouds = []
    contact_patches = []
    contact_labels = []
    points_iou = []
    occ_iou = []

    # Load ground truth meshes and surface contact labels.
    for idx in range(n):
        dataset_dict = dataset[idx]
        meshes.append(dataset.get_example_mesh(idx))
        pointclouds.append(torch.from_numpy(dataset_dict["surface_points"]).to(device))

        # Load contact patch.
        contact_patches.append(torch.from_numpy(dataset_dict["contact_patch"]).to(device))

        contact_labels.append(torch.from_numpy(dataset_dict["surface_in_contact"]).to(device).int())
        points_iou.append(torch.from_numpy(dataset_dict["points_iou"]).to(device))
        occ_iou.append(torch.from_numpy(dataset_dict["occ_tgt"]).to(device).int())

    return meshes, pointclouds, contact_patches, contact_labels, points_iou, occ_iou


def load_gt_results_real(dataset, dataset_dir, n, device=None):
    contact_patches = []

    # Load ground truth meshes and surface contact labels.
    for idx in range(n):
        dataset_dict = dataset[idx]

        # Load contact patch.
        contact_patch = torch.from_numpy(dataset_dict["contact_patch"]).to(device).float()
        contact_patches.append(contact_patch)

    return contact_patches


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
