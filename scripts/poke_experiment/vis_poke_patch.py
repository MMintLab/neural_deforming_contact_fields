import argparse
import os

import trimesh
from vedo import Plotter, Mesh, Points, Video

import mmint_utils
from neural_contact_fields.utils import utils
from scripts.video.vedo_animator import VedoOrbiter


def vis_poke_patch(gen_dir: str, idx: int, pitch: float = 0.0, no_animation: bool = False):
    # Load dataset.
    # dataset_cfg, dataset = load_dataset_from_config(dataset_cfg_fn)
    # data_dict = dataset[idx]

    # Load mesh.
    obj_mesh = trimesh.load(os.path.join(gen_dir, "nontextured.stl"))

    # Load ground truth pose.
    gt_pose = mmint_utils.load_gzip_pickle(os.path.join(gen_dir, "gt_icp.pkl.gzip"))[0]["transformation"]

    # Apply transform to the mesh.
    obj_mesh.apply_transform(gt_pose)

    # Load predicted sponge mesh.
    mesh = trimesh.load(os.path.join(gen_dir, "mesh_w_%d.obj" % idx))

    # Load contact patch.
    patch = utils.load_pointcloud(os.path.join(gen_dir, "contact_patch_w_%d.ply" % idx))

    # Build plotter.
    things_to_show = [
        Mesh([mesh.vertices, mesh.faces], c="gold", alpha=1.0),
        Points(patch[:, :3], c="r"),
        Mesh([obj_mesh.vertices, obj_mesh.faces], c="grey", alpha=1.0)
    ]
    plot = Plotter(shape=(1, 1), interactive=False)
    plot.at(0).show(*things_to_show)

    if no_animation:
        # First - highlight reconstruction.
        things_to_show[2].alpha(0.2)
        things_to_show[1].alpha(0.0)
        plot.interactive().close()

        # Second - highlight contact patch.
        things_to_show[2].alpha(0.2)
        things_to_show[1].alpha(1.0)
        things_to_show[0].alpha(0.2)
        plot = Plotter(shape=(1, 1), interactive=False)
        plot.at(0).show(*things_to_show)
        plot.interactive().close()

        # Third - reconstruction, no object.
        things_to_show[0].alpha(1.0)
        things_to_show[1].alpha(0.0)
        things_to_show[2].alpha(0.0)
        plot = Plotter(shape=(1, 1), interactive=False)
        plot.at(0).show(*things_to_show)
        plot.interactive().close()

        # Fourth - patch, no object.
        things_to_show[0].alpha(0.2)
        things_to_show[1].alpha(1.0)
        things_to_show[2].alpha(0.0)
        plot = Plotter(shape=(1, 1), interactive=False)
        plot.at(0).show(*things_to_show)
        plot.interactive().close()

        # Fifth - partial pointcloud.
        plot = Plotter(shape=(1, 1), interactive=False)
        # plot.at(0).show(Points(data_dict["partial_pointcloud"][:, :3], c="black"))
        plot.interactive().close()
    else:
        orbiter = VedoOrbiter(plot, things_to_show, period=5, dist=0.4, pitch=pitch, target=obj_mesh.centroid)

        fps = int(1.0 / orbiter.update_period)
        num_per_spin = int(fps * orbiter.period)

        # video = Video(os.path.join(gen_dir, "vis_%d.mp4" % idx), backend="ffmpeg", fps=fps)

        for spin_idx in range(num_per_spin):
            orbiter.update_transform(orbiter.generate_transform(orbiter.update_period * spin_idx))

            # video.add_frame()

        for spin_idx in range(num_per_spin):
            orbiter.update_transform(orbiter.generate_transform(orbiter.update_period * spin_idx))
            things_to_show[2].alpha(max(0.2, 1.0 - (spin_idx / (num_per_spin / 10.0)) * 0.8))

            # sleep(0.01)
            # video.add_frame()

        for spin_idx in range(num_per_spin):
            orbiter.update_transform(orbiter.generate_transform(orbiter.update_period * spin_idx))
            things_to_show[2].alpha(min(0.3, 0.2 + (spin_idx / (num_per_spin / 10.0)) * 0.1))
            things_to_show[0].alpha(max(0.2, 1.0 - (spin_idx / (num_per_spin / 10.0)) * 0.8))

            # sleep(0.01)
            # video.add_frame()

        for spin_idx in range(num_per_spin):
            orbiter.update_transform(orbiter.generate_transform(orbiter.update_period * spin_idx))
            things_to_show[2].alpha(min(1.0, 0.3 + (spin_idx / (num_per_spin / 10.0)) * 0.7))
            things_to_show[0].alpha(min(1.0, 0.2 + (spin_idx / (num_per_spin / 10.0)) * 0.8))

            # video.add_frame()

        # video.close()
        plot.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate contact patches.")
    parser.add_argument("gen_dir", type=str, help="Directory with contact patches.")
    parser.add_argument("idx", type=int, help="Index of example to visualize.")
    parser.add_argument("--pitch", type=float, default=0.5, help="Pitch of camera.")
    parser.add_argument("--no_animation", action="store_true", help="Don't animate.")
    parser.set_defaults(no_animation=False)
    args = parser.parse_args()

    vis_poke_patch(args.gen_dir, args.idx, args.pitch,
                   args.no_animation)
