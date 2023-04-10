import argparse
import copy
import os
from typing import List

import numpy as np
import open3d as o3d
from tqdm import trange

import mmint_utils
import pytorch_kinematics as pk


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def icp_gt(mesh_fn: str, pointclouds: List[str], out_fn: str):
    N_icp = 100

    # Load mesh and sample point cloud.
    mesh = o3d.io.read_triangle_mesh(mesh_fn)
    mesh_pcd = mesh.sample_points_poisson_disk(10000)
    # mesh_pcd = o3d.io.read_point_cloud(mesh_fn)

    # Load pointclouds.
    pcds = [o3d.io.read_point_cloud(pcd_fn) for pcd_fn in pointclouds]

    # Combine pointclouds.
    target_pcd = o3d.geometry.PointCloud()
    for pcd in pcds:
        target_pcd += pcd

    # Sample many initial orientations.
    tf_init = pk.random_rotations(100).cpu().numpy()
    results = []

    for icp_idx in trange(N_icp):
        # Transform initialization.
        run_init = np.eye(4)
        run_init[:3, :3] = tf_init[icp_idx]
        run_init[:3, 3] = target_pcd.get_center() - mesh_pcd.get_center()

        # Visualize initial guess.
        # draw_registration_result(mesh_pcd, target_pcd, tf_init)

        # Run ICP.
        reg_p2p = o3d.pipelines.registration.registration_icp(
            mesh_pcd, target_pcd, 0.01, run_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))
        # print(reg_p2p)

        results.append(reg_p2p)

    # Sort based on fit.
    results = sorted(results, key=lambda x: x.fitness, reverse=True)

    # Visualize result.
    # for res in results:
    #     draw_registration_result(mesh_pcd, target_pcd, res.transformation)
    draw_registration_result(mesh_pcd, target_pcd, results[0].transformation)

    results_to_save = [{
        "fitness": res.fitness,
        "inlier_rmse": res.inlier_rmse,
        "transformation": res.transformation,
    } for res in results]

    # Save result.
    mmint_utils.make_dir(os.path.dirname(out_fn))
    mmint_utils.save_gzip_pickle(results_to_save, out_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ICP with setup pointclouds to get approx. GT.")
    parser.add_argument("mesh_fn", type=str, help="Mesh to align to.")
    parser.add_argument("pointclouds", type=str, nargs="+", help="Pointclouds to align.")
    parser.add_argument("out_fn", type=str, help="Output file.")
    args = parser.parse_args()

    icp_gt(args.mesh_fn, args.pointclouds, args.out_fn)
