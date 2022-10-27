import numpy as np
from neural_contact_fields.vis import plot_points
from neural_contact_fields.data.dataset_helpers import load_dataset_dict

if __name__ == '__main__':
    dataset_file = "/home/markvdm/Documents/NCF/VIRDO/data/virdo_simul_dataset.pickle"

    dataset_dict = load_dataset_dict(dataset_file)

    train_data = dataset_dict["train"]

    for train_key in [0]:  # train_data.keys():
        tool_data = train_data[train_key]
        for deform_key in [0]:  # tool_data.keys():
            data_dict = tool_data[deform_key]

            coords = data_dict["coords"][0]
            sdf_gt = data_dict["gt"][0]

            surface_points = coords[(sdf_gt == 0.0).view(-1)]
            colors = np.zeros([surface_points.shape[0], 3], dtype=np.float32)
            colors[:, 2] = 1.0

            if "contact" in data_dict:
                contact_points = data_dict["contact"][0]
                contact_colors = np.zeros([contact_points.shape[0], 3], dtype=np.float32)
                contact_colors[:, 0] = 1.0

                surface_points = np.concatenate([surface_points, contact_points], axis=0)
                colors = np.concatenate([colors, contact_colors], axis=0)

            plot_points(surface_points, colors=colors)
