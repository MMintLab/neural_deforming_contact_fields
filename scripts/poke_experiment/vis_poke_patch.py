import argparse
import os
from time import sleep

import trimesh
from vedo import Plotter, Mesh, Points, Video

import mmint_utils
from neural_contact_fields.utils import utils
from scripts.video.vedo_animator import VedoOrbiter


def vis_poke_patch(gen_dir: str, mesh_fn: str, gt_pose_fn: str, idx: int, pitch: float = 0.0):
    # Load mesh.
    obj_mesh = trimesh.load(mesh_fn)

    # Load ground truth pose.
    gt_pose = mmint_utils.load_gzip_pickle(gt_pose_fn)[0]["transformation"]

    # Apply transform to the mesh.
    obj_mesh.apply_transform(gt_pose)

    # Load predicted sponge mesh.
    mesh = trimesh.load(os.path.join(gen_dir, "mesh_w_%d.obj" % idx))

    # Load contact patch.
    patch = utils.load_pointcloud(os.path.join(gen_dir, "contact_patch_w_%d.ply" % idx))

    # Build plotter.
    plot = Plotter(shape=(1, 1), interactive=False)
    things_to_show = [
        Mesh([mesh.vertices, mesh.faces], c="gold", alpha=1.0),
        Points(patch[:, :3], c="r"),
        Mesh([obj_mesh.vertices, obj_mesh.faces], c="grey", alpha=1.0)
    ]
    plot.at(0).show(*things_to_show)

    orbiter = VedoOrbiter(plot, things_to_show, period=5, dist=0.4, pitch=pitch, target=obj_mesh.centroid)

    fps = int(1.0 / orbiter.update_period)
    num_per_spin = int(fps * orbiter.period)

    video = Video(os.path.join(gen_dir, "vis_%d.mp4" % idx), backend="ffmpeg", fps=fps)

    for spin_idx in range(num_per_spin):
        orbiter.update_transform(orbiter.generate_transform(orbiter.update_period * spin_idx))

        video.add_frame()

    for spin_idx in range(num_per_spin):
        orbiter.update_transform(orbiter.generate_transform(orbiter.update_period * spin_idx))
        things_to_show[2].alpha(max(0.2, 1.0 - (spin_idx / (num_per_spin / 10.0)) * 0.8))

        # sleep(0.01)
        video.add_frame()

    for spin_idx in range(num_per_spin):
        orbiter.update_transform(orbiter.generate_transform(orbiter.update_period * spin_idx))
        things_to_show[2].alpha(min(0.3, 0.2 + (spin_idx / (num_per_spin / 10.0)) * 0.1))
        things_to_show[0].alpha(max(0.2, 1.0 - (spin_idx / (num_per_spin / 10.0)) * 0.8))

        # sleep(0.01)
        video.add_frame()

    for spin_idx in range(num_per_spin):
        orbiter.update_transform(orbiter.generate_transform(orbiter.update_period * spin_idx))
        things_to_show[2].alpha(min(1.0, 0.3 + (spin_idx / (num_per_spin / 10.0)) * 0.7))
        things_to_show[0].alpha(min(1.0, 0.2 + (spin_idx / (num_per_spin / 10.0)) * 0.8))

        video.add_frame()

    video.close()
    plot.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate contact patches.")
    parser.add_argument("gen_dir", type=str, help="Directory with contact patches.")
    parser.add_argument("mesh_fn", type=str, help="Mesh to align to.")
    parser.add_argument("gt_pose_fn", type=str, help="Ground truth pose.")
    parser.add_argument("idx", type=int, help="Index of example to visualize.")
    parser.add_argument("--pitch", type=float, default=0.5, help="Pitch of camera.")
    args = parser.parse_args()

    vis_poke_patch(args.gen_dir, args.mesh_fn, args.gt_pose_fn, args.idx, args.pitch)
