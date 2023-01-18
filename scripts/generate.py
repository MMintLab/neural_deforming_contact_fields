import os

import mmint_utils
from neural_contact_fields import config
from neural_contact_fields.utils import utils
from neural_contact_fields.utils.args_utils import get_model_dataset_arg_parser, load_model_dataset_from_args
from tqdm import trange


def generate(model_cfg, model, dataset, device, out_dir):
    model.eval()

    # Load generator.
    generator = config.get_generator(model_cfg, model, device)

    # Determine what to generate.
    generate_mesh = generator.generates_mesh
    generate_pointcloud = generator.generates_pointcloud
    generate_contact_patch = generator.generates_contact_patch
    generate_contact_labels = generator.generates_contact_labels

    # Create output directory.
    if out_dir is not None:
        mmint_utils.make_dir(out_dir)

    # Go through dataset and generate!
    for idx in trange(len(dataset)):
        data_dict = dataset[idx]
        metadata = {}

        if generate_mesh:
            mesh, metadata_mesh = generator.generate_mesh(data_dict, metadata)
            metadata = mmint_utils.combine_dict(metadata, metadata_mesh)

            # Write mesh to file.
            if out_dir is not None:
                mesh_fn = os.path.join(out_dir, "mesh_%d.obj" % idx)
                mesh.export(mesh_fn)

        if generate_pointcloud:
            pointcloud, metadata_pc = generator.generate_pointcloud(data_dict, metadata)
            metadata = mmint_utils.combine_dict(metadata, metadata_pc)

            # Write pointcloud to file.
            if out_dir is not None:
                pc_fn = os.path.join(out_dir, "pointcloud_%d.ply" % idx)
                utils.save_pointcloud(pointcloud, pc_fn)

        if generate_contact_patch:
            contact_patch, metadata_cp = generator.generate_contact_patch(data_dict, metadata)
            metadata = mmint_utils.combine_dict(metadata, metadata_cp)

            # Write contact patch to file (as pointcloud). TODO: Is this the write representation?
            if out_dir is not None:
                cp_fn = os.path.join(out_dir, "contact_patch_%d.ply" % idx)
                utils.save_pointcloud(contact_patch, cp_fn)

        if generate_contact_labels:
            contact_labels, metadata_cl = generator.generate_contact_labels(data_dict, metadata)

            # Write contact labels to file.
            if out_dir is not None:
                cl_fn = os.path.join(out_dir, "contact_labels_%d.pkl.gzip" % idx)
                mmint_utils.save_gzip_pickle(contact_labels, cl_fn)


if __name__ == '__main__':
    parser = get_model_dataset_arg_parser()
    parser.add_argument("--out", "-o", type=str, help="Optional out directory to write generated results to.")
    # TODO: Add visualization?
    args = parser.parse_args()

    model_cfg_, model_, dataset_, device_ = load_model_dataset_from_args(args)
    generate(model_cfg_, model_, dataset_, device_, args.out)
