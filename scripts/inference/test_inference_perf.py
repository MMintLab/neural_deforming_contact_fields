import argparse
import os

import mmint_utils
import trimesh
from neural_contact_fields.data.tool_dataset import ToolDataset
from neural_contact_fields.inference import infer_latent_from_surface
from neural_contact_fields.utils import vedo_utils
from neural_contact_fields.utils.model_utils import load_model_and_dataset
from vedo import Plotter, Mesh


def get_model_dataset_arg_parser():
    """
    Argument parser for common model + dataset arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Model/data config file.")
    parser.add_argument("--out", "-o", type=str, default=None, help="Directory to write results to.")
    parser.add_argument("--dataset_config", "-d", type=str, default=None, help="Optional dataset config to use.")
    parser.add_argument("--mode", "-m", type=str, default="test", help="Which split to vis [train, val, test].")
    parser.add_argument("--model_file", "-f", type=str, default="model.pt", help="Which model save file to use.")
    parser.add_argument("-v", "--vis", dest="vis", action="store_true", help="Visualize.")
    parser.set_defaults(vis=False)
    return parser


def load_model_dataset_from_args(args):
    """
    Load model and dataset from arguments object.
    """
    model_cfg, model, dataset, device = load_model_and_dataset(args.config, dataset_config=args.dataset_config,
                                                               dataset_mode=args.mode,
                                                               model_file=args.model_file)
    model.eval()
    return model_cfg, model, dataset, device


def vis_mesh_prediction(pred_mesh: trimesh.Trimesh, gt_mesh: trimesh.Trimesh):
    plt = Plotter(shape=(1, 2))
    plt.at(0).show(Mesh([gt_mesh.vertices, gt_mesh.faces]), vedo_utils.draw_origin(), "GT Mesh")
    plt.at(1).show(Mesh([pred_mesh.vertices, pred_mesh.faces]), vedo_utils.draw_origin(), "Pred. Mesh")
    plt.interactive().close()


def test_inference_perf(args):
    dataset: ToolDataset
    model_cfg, model, dataset, device = load_model_dataset_from_args(args)

    vis = args.vis

    # Load meshes.
    gt_meshes = []
    dataset_dir = dataset.dataset_dir
    for trial_idx in range(len(dataset)):
        mesh_fn = os.path.join(dataset_dir, "out_%d_mesh.obj" % trial_idx)
        mesh = trimesh.load(mesh_fn)
        gt_meshes.append(mesh)

    out_dir = args.out
    if out_dir is not None:
        mmint_utils.make_dir(out_dir)

    for trial_idx in range(len(dataset)):
        trial_dict = dataset[trial_idx]
        latent_code, pred_dict, surface_pred_dict, mesh = infer_latent_from_surface(model, trial_dict, {},
                                                                                    device=device)

        # Compare meshes.
        if vis:
            vis_mesh_prediction(mesh, gt_meshes[trial_idx])

        # Write results.
        if out_dir is not None:
            # Write predicted mesh to file.
            mesh.export(os.path.join(out_dir, "pred_%d_mesh.obj" % trial_idx))

            # Write surface predictions to file.
            mmint_utils.save_gzip_pickle(surface_pred_dict,
                                         os.path.join(out_dir, "pred_%d_surface.pkl.gzip" % trial_idx))


if __name__ == '__main__':
    parser_ = get_model_dataset_arg_parser()
    args_ = parser_.parse_args()
    test_inference_perf(args_)
