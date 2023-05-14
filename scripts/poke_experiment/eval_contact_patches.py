import argparse
import os

import numpy as np
import open3d as o3d

import mmint_utils


def eval_contact_patches(gen_dir: str, mesh_fn: str, gt_pose_fn: str, out_fn: str):
    # Load contact patches from generation directory.
    contact_patches_fns = [f for f in os.listdir(gen_dir) if "contact_patch_w_" in f]

    # Load contact patches.
    contact_patches = [o3d.io.read_point_cloud(os.path.join(gen_dir, f)) for f in contact_patches_fns]

    # Combine contact patches.
    contact_patches_combined = o3d.geometry.PointCloud()
    for cp in contact_patches:
        contact_patches_combined += cp

    # Load mesh.
    mesh = o3d.io.read_triangle_mesh(mesh_fn)

    # Get ground truth pose that was precomputed.
    gt_pose = mmint_utils.load_gzip_pickle(gt_pose_fn)[0]["transformation"]

    # Apply transform to the mesh.
    mesh.transform(gt_pose)

    # Visualize mesh and combined pointcloud.
    o3d.visualization.draw_geometries([mesh, contact_patches_combined])

    # Setup distance checks.
    mesh_ = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_)

    # Compute distances.
    query_points = o3d.core.Tensor(np.asarray(contact_patches_combined.points), dtype=o3d.core.Dtype.Float32)
    unsigned_distance = scene.compute_distance(query_points)

    # Compute squared distance.
    squared_distance = unsigned_distance.numpy()
    score = (squared_distance ** 2.0).mean() * 1e6

    # Save score.
    mmint_utils.save_gzip_pickle(
        {"score": score}, out_fn
    )
    print("Score: %f" % score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate contact patches.")
    parser.add_argument("gen_dir", type=str, help="Directory with contact patches.")
    parser.add_argument("mesh_fn", type=str, help="Mesh to align to.")
    parser.add_argument("gt_pose_fn", type=str, help="Ground truth pose.")
    parser.add_argument("out_fn", type=str, help="Output file.")
    args = parser.parse_args()

    eval_contact_patches(args.gen_dir, args.mesh_fn, args.gt_pose_fn, args.out_fn)
