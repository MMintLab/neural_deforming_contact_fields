import argparse

import numpy as np
from matplotlib import pyplot as plt

from neural_contact_fields.utils.model_utils import load_dataset_from_config


def wrench_statistics(dataset):
    print("Dataset size: %d" % len(dataset))

    wrenches = []
    wrench_norms = []
    for data_idx in range(len(dataset)):
        data_dict = dataset[data_idx]

        wrenches.append(data_dict["wrist_wrench"])
        wrench_norms.append(np.linalg.norm(data_dict["wrist_wrench"]))
    wrenches = np.stack(wrenches)

    # Create a matplotlib histogram for each dimension.
    fig, axs = plt.subplots(2, 4, sharey=True, tight_layout=True)
    for dim, dim_name in zip(range(6), ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]):
        axs[dim // 3, dim % 3].hist(wrenches[:, dim], bins=20)
        axs[dim // 3, dim % 3].set_title(dim_name)
    axs[1, 3].hist(wrench_norms, bins=20)
    axs[1, 3].set_title("Norm")

    # Visualize the wrench of an individual example.
    idx = 36
    data_dict = dataset[idx]
    wrench = data_dict["wrist_wrench"]
    wrench_norm = np.linalg.norm(wrench)
    for dim, dim_name in zip(range(6), ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]):
        axs[dim // 3, dim % 3].axvline(wrench[dim], color="red")
    axs[1, 3].axvline(wrench_norm, color="red")

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize dataset.")
    parser.add_argument("dataset_cfg", type=str, help="Path to dataset config.")
    parser.add_argument("--mode", "-m", type=str, default="test", help="Which dataset split to use [train, val, test].")
    args = parser.parse_args()

    _, dataset_ = load_dataset_from_config(args.dataset_cfg, dataset_mode=args.mode)
    wrench_statistics(dataset_)
