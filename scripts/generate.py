import os

import open3d as o3d
import mmint_utils
from neural_contact_fields import config
from neural_contact_fields.utils import utils
from neural_contact_fields.utils.args_utils import get_model_dataset_arg_parser, load_model_dataset_from_args
from neural_contact_fields.utils.results_utils import write_results
from tqdm import trange
import time
import numpy as np

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
    cd_hist = {'pcd': [] , 'pcd_cnt': []}
    time_hist = {'inference': [], 'meshing': []}
    for idx in trange(len(dataset)):
        data_dict = dataset[idx]
        metadata = {}
        mesh = pointcloud = contact_patch = contact_labels = None


        if generate_pointcloud:
            start = time.time()
            pointcloud, cd = generator.generate_pointcloud(data_dict, metadata)
            cd_hist['pcd'].append(cd)
            time_hist['inference'].append(time.time()-start)

        if generate_mesh:
            mesh, metadata_mesh = generator.generate_mesh(data_dict, metadata)
            metadata = mmint_utils.combine_dict(metadata, metadata_mesh)



        else:
            # Generate mesh from the pointcloud.
            start = time.time()
            mesh = generator.generate_mesh_from_pointcloud(pointcloud)
            time_hist['meshing'].append(time.time() - start)


        if generate_contact_patch:
            contact_patch, cd = generator.generate_contact_patch(data_dict, metadata)
            cd_hist['pcd_cnt'].append(cd)

        if generate_contact_labels:
            contact_labels, metadata_cl = generator.generate_contact_labels(data_dict, metadata)
        write_results(out_dir, mesh, pointcloud, contact_patch, contact_labels, idx)


    print(time_hist)
    print("summary", np.mean(time_hist['inference']), np.std(time_hist['inference']), np.mean(time_hist['meshing']), np.std(time_hist['meshing']))


if __name__ == '__main__':
    parser = get_model_dataset_arg_parser()
    parser.add_argument("--out", "-o", type=str, help="Optional out directory to write generated results to.")
    # TODO: Add visualization?
    args = parser.parse_args()

    model_cfg_, model_, dataset_, device_ = load_model_dataset_from_args(args)
    generate(model_cfg_, model_, dataset_, device_, args.out)
