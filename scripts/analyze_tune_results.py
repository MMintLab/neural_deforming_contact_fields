import matplotlib.pyplot as plt
import numpy as np
from ray import tune
import argparse


def get_matching_result(results_gird: tune.ResultGrid, config: dict):
    for result in results_gird:
        if result.config == config:
            return result

    raise RuntimeError("No matching result found.")


def analyze_tune_results(results_dir: str):
    restored_tuner = tune.Tuner.restore(results_dir)

    result_grid: tune.ResultGrid = restored_tuner.get_results()
    best_result = result_grid.get_best_result(metric="patch_chamfer_distance_mean", mode="min")

    print("Best config: ", best_result.config)
    print("Best metric: ", best_result.metrics["patch_chamfer_distance_mean"])

    # Grid visualization of configs and their effect on performance.

    # Define our search space. TODO: Load this from a config file.
    search_space = {
        "contact_threshold": [0.2, 0.5, 0.8],
        "embed_weight": [1e-3, 1e-1, 1.0],
        "iter_limit": [100, 300, 500, 1000],
    }

    # Create grids based on embed_weight and iter_limit with a fixed contact_threshold.
    for contact_threshold in search_space["contact_threshold"]:
        grid = np.zeros((len(search_space["embed_weight"]), len(search_space["iter_limit"])))

        for embed_idx, embed_weight in enumerate(search_space["embed_weight"]):
            for iter_idx, iter_limit in enumerate(search_space["iter_limit"]):
                config = {
                    "contact_threshold": contact_threshold,
                    "embed_weight": embed_weight,
                    "iter_limit": iter_limit,
                }

                result = get_matching_result(result_grid, config)
                grid[embed_idx, iter_idx] = result.metrics["patch_chamfer_distance_mean"]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        x_ticks = np.arange(len(search_space["iter_limit"]))
        y_ticks = np.arange(len(search_space["embed_weight"]))
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels(search_space["iter_limit"])
        ax.set_yticklabels(search_space["embed_weight"])
        ax.set_ylabel("embed_weight")
        ax.set_xlabel("iter_limit")
        ax.set_title(f"contact_threshold={contact_threshold}")
        grid_vis = ax.matshow(grid)
        fig.colorbar(grid_vis)

        for (i, j), m in np.ndenumerate(grid):
            ax.text(j, i, f"{m:.2f}", ha="center", va="center")

        plt.show()
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("tune_dir", type=str, help="Directory with tune results.")
    args = parser.parse_args()

    analyze_tune_results(results_dir=args.tune_dir)
