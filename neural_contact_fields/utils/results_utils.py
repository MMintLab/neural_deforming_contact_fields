import os

import mmint_utils
import trimesh
from neural_contact_fields.utils import utils


def write_results(out_dir, mesh, pointcloud, contact_patch, contact_labels, idx):
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


def load_pred_results(out_dir, n):
    meshes = []
    pointclouds = []
    contact_patches = []
    contact_labels = []

    for idx in range(n):
        mesh_fn = os.path.join(out_dir, "mesh_%d.obj" % idx)
        if os.path.exists(mesh_fn):
            meshes.append(trimesh.load(mesh_fn))
        else:
            meshes.append(None)

        pc_fn = os.path.join(out_dir, "pointcloud_%d.ply" % idx)
        if os.path.exists(pc_fn):
            pointclouds.append(utils.load_pointcloud(pc_fn))
        else:
            pointclouds.append(None)

        cp_fn = os.path.join(out_dir, "contact_patch_%d.ply" % idx)
        if os.path.exists(cp_fn):
            contact_patches.append(utils.load_pointcloud(cp_fn))
        else:
            contact_patches.append(None)

        cl_fn = os.path.join(out_dir, "contact_labels_%d.pkl.gzip" % idx)
        if os.path.exists(cl_fn):
            contact_labels.append(mmint_utils.load_gzip_pickle(cl_fn))
        else:
            contact_labels.append(None)

    return meshes, pointclouds, contact_patches, contact_labels


def load_gt_results(dataset, dataset_dir, n):
    meshes = []
    pointclouds = []
    contact_patches = []
    contact_labels = []
    points_iou = []
    occ_iou = []

    # Load ground truth meshes and surface contact labels.
    for idx in range(n):
        data_dict = mmint_utils.load_gzip_pickle(os.path.join(dataset_dir, "out_%d.pkl.gzip" % idx))
        meshes.append(trimesh.load(os.path.join(dataset_dir, "out_%d_mesh.obj" % idx)))
        pointclouds.append(None)
        contact_patches.append(None)
        contact_labels.append(dataset[idx]["surface_in_contact"])
        points_iou.append(data_dict["test"]["points_iou"])
        occ_iou.append(data_dict["test"]["occ_tgt"])

    return meshes, pointclouds, contact_patches, contact_labels, points_iou, occ_iou
