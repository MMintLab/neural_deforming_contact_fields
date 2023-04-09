import os
import argparse
import pdb

import open3d as o3d


def setup_mesh_pc(mesh_fn: str):
    # Load mesh and sample point cloud.
    mesh = o3d.io.read_triangle_mesh(mesh_fn)
    mesh_pcd = mesh.sample_points_poisson_disk(10000)

    # Prune out points with z > 0.1 and that are within 0.035 of xy origin.
    keep_idcs = [i for i, p in enumerate(mesh_pcd.points) if
                 not (p[2] > 0.002 and ((p[0] + 0.02) ** 2 + (p[1] - 0.02) ** 2) < 0.035 ** 2)]
    mesh_pcd = mesh_pcd.select_by_index(keep_idcs)

    # Save pointcloud.
    o3d.io.write_point_cloud(os.path.join(os.path.dirname(mesh_fn), "mesh.ply"), mesh_pcd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Setup mesh point cloud.")
    parser.add_argument("mesh_fn", type=str, help="Mesh file name.")
    args = parser.parse_args()

    setup_mesh_pc(args.mesh_fn)
