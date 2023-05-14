from typing import List

import matplotlib.pyplot as plt
import numpy as np
from ray import tune
import argparse


def get_matching_result(results_grids: List[tune.ResultGrid], config: dict):
    for result_grid in results_grids:
        for result in result_grid:
            if result.config == config:
                return result

    raise RuntimeError("No matching result found for config ", config)


def analyze_tune_results(results_dirs: str):
    restored_tuners = [tune.Tuner.restore(results_dir) for results_dir in results_dirs]

    result_grids = [restored_tuner.get_results() for restored_tuner in restored_tuners]

    # Get best result from all tuners.
    best_results = [result_grid.get_best_result(metric="patch_chamfer_distance_mean", mode="min") for result_grid in
                    result_grids]
    best_result = min(best_results, key=lambda result_: result_.metrics["patch_chamfer_distance_mean"])
    print("Best config: ", best_result.config)
    print("Best metric: ", best_result.metrics["patch_chamfer_distance_mean"])

    # Grid visualization of configs and their effect on performance.
    # metrics = ["patch_chamfer_distance", "patch_percent", "chamfer_distance", "iou"]
    # metrics_mode = ["min", "max", "min", "max"]
    # has_var = [True, False, True, True]
    # metrics = ["patch_chamfer_distance", "chamfer_distance", "iou"]
    # metrics_mode = ["min", "min", "max"]
    # has_var = [True, True, True]
    metrics = ["patch_chamfer_distance", "patch_percent"]
    metrics_mode = ["min", "max"]
    has_var = [True, False]

    # Define our search space. TODO: Load this from a config file.
    search_space = {
        "contact_threshold": [0.2, 0.5, 0.8],
        "embed_weight": [1e-3, 1e-1, 1.0],
        "iter_limit": [50, 100, 300],
    }

    # Create grids based on embed_weight and iter_limit with a fixed contact_threshold.
    for contact_threshold in search_space["contact_threshold"]:
        metrics_dict = dict()
        for m_idx, metric in enumerate(metrics):
            if has_var[m_idx]:
                metrics_dict[f"{metric}_mean"] = np.zeros(
                    (len(search_space["embed_weight"]), len(search_space["iter_limit"])))
                metrics_dict[f"{metric}_std"] = np.zeros(
                    (len(search_space["embed_weight"]), len(search_space["iter_limit"])))
            else:
                metrics_dict[metric] = np.zeros(
                    (len(search_space["embed_weight"]), len(search_space["iter_limit"])))

        for embed_idx, embed_weight in enumerate(search_space["embed_weight"]):
            for iter_idx, iter_limit in enumerate(search_space["iter_limit"]):
                config = {
                    "contact_threshold": contact_threshold,
                    "embed_weight": embed_weight,
                    "iter_limit": iter_limit,
                }

                result = get_matching_result(result_grids, config)

                for m_idx, metric in enumerate(metrics):
                    if has_var[m_idx]:
                        metrics_dict[f"{metric}_mean"][embed_idx, iter_idx] = result.metrics[f"{metric}_mean"]
                        metrics_dict[f"{metric}_std"][embed_idx, iter_idx] = result.metrics[f"{metric}_std"]
                    else:
                        metrics_dict[metric][embed_idx, iter_idx] = result.metrics[metric]

        fig = plt.figure()
        for m_idx, metric in enumerate(metrics):
            ax = fig.add_subplot(1, len(metrics), m_idx + 1)
            x_ticks = np.arange(len(search_space["iter_limit"]))
            y_ticks = np.arange(len(search_space["embed_weight"]))
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.set_xticklabels(search_space["iter_limit"])
            ax.set_yticklabels(search_space["embed_weight"])
            ax.set_ylabel("embed_weight")
            ax.set_xlabel("iter_limit")
            ax.set_title(metric)

            # print(metrics_dict[f"{metric}_mean"])
            # print(metrics_dict[f"{metric}_std"])

            grid_vis = ax.matshow(metrics_dict[f"{metric}_mean"] if has_var[m_idx] else metrics_dict[metric],
                                  cmap="viridis" if metrics_mode[m_idx] == "max" else "viridis_r")
            fig.colorbar(grid_vis)

            if has_var[m_idx]:
                for (i, j), m in np.ndenumerate(metrics_dict[f"{metric}_mean"]):
                    std = metrics_dict[f"{metric}_std"][i, j]
                    ax.text(j, i, f"{m:.3f}\n({std:.3f})", ha="center", va="center")
            else:
                for (i, j), m in np.ndenumerate(metrics_dict[metric]):
                    ax.text(j, i, f"{m:.3f}", ha="center", va="center")

        fig.suptitle(f"contact_threshold={contact_threshold}")
        plt.show()
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("tune_dirs", nargs="+", type=str, help="Directories with tune results.")
    args = parser.parse_args()

    analyze_tune_results(results_dirs=args.tune_dirs)
