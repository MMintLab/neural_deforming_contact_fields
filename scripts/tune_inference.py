import yaml
from tqdm import trange
import os

import mmint_utils
from neural_contact_fields import config
from neural_contact_fields.utils.args_utils import get_model_dataset_arg_parser, load_model_dataset_from_args
from neural_contact_fields.utils.model_utils import load_generation_cfg
from neural_contact_fields.utils.results_utils import load_gt_results, metrics_to_statistics, write_results

from ray.air import session
from ray import tune, air
from ray.tune.search.bayesopt import BayesOptSearch

from scripts.eval_results import eval_example
from scripts.generate import generate_example


def tune_inference(args):
    out_dir = args.out
    search_alg = args.search_alg
    restore = args.restore

    # Set visible gpus to the one provided.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(c_id) for c_id in args.cuda_ids])
    args.cuda_id = 0  # Change index to be into visible devices list.

    # Make out dir.
    mmint_utils.make_dir(out_dir)

    # Define gen/eval function use by RayTune.
    def eval_hyper_params(param_config: dict):
        model_cfg, model, dataset, device = load_model_dataset_from_args(args)
        model.eval()

        # Load generate cfg, if present.
        base_generation_cfg = load_generation_cfg(model_cfg, args.model_file)
        if args.gen_args is not None:
            base_generation_cfg.update(args.gen_args)

        # Load ground truth results.
        gt_dicts = load_gt_results(dataset, len(dataset), device)

        gen_cfg = mmint_utils.combine_dict(base_generation_cfg, param_config)

        # Load generator.
        generator = config.get_generator(model_cfg, model, gen_cfg, device)

        # Go through dataset and generate!
        metrics_dicts = []
        for idx in range(len(dataset)):
            data_dict = dataset[idx]

            gen_dict = generate_example(generator, data_dict)
            write_results("./", gen_dict, idx)

            gt_dict = gt_dicts[idx]

            # Evaluate.
            metrics_dict = eval_example(gen_dict, gt_dict, device)
            metrics_dicts.append(metrics_dict)
        metrics_dicts = [m["metrics"] for m in metrics_dicts]

        # Compute stats on metrics.
        metrics_stats_dict = metrics_to_statistics(metrics_dicts)
        session.report(metrics_stats_dict)

    trainable_with_resources = tune.with_resources(eval_hyper_params, {"cpu": 16, "gpu": 1})

    if restore:
        # Restore tuner.
        tuner = tune.Tuner.restore(
            os.path.join(out_dir, search_alg),
            trainable_with_resources,
            resume_errored=False, restart_errored=True
        )
    else:
        # Define search space. TODO: Move this to a config file.
        search_space = {
            "contact_threshold": tune.quniform(0.0, 1.0, 0.1),
            "embed_weight": tune.loguniform(1e-6, 1.0),
            "iter_limit": tune.qrandint(100, 1000, 100),
        }

        # Setup search algorithm.
        if search_alg == "bayes":
            search_space["iter_limit"] = tune.uniform(100, 1000)  # Bayes doesn't support int search spaces.
            search_alg_ = BayesOptSearch(metric="patch_chamfer_distance_mean", mode="min", random_search_steps=4)
            tune_cfg = tune.TuneConfig(search_alg=search_alg_, num_samples=-1)
        elif search_alg == "grid":
            tune_cfg = tune.TuneConfig(metric="patch_chamfer_distance_mean", mode="min")
            search_space["contact_threshold"] = tune.grid_search([0.2, 0.5, 0.8])
            search_space["embed_weight"] = tune.grid_search([1.0])
            search_space["iter_limit"] = tune.grid_search([50, 100, 300])
        else:
            tune_cfg = tune.TuneConfig(metric="patch_chamfer_distance_mean", mode="min", num_samples=10)

        # Setup and run tuner.
        tuner = tune.Tuner(
            trainable_with_resources,
            tune_config=tune_cfg,
            run_config=air.RunConfig(local_dir=out_dir, name=search_alg),
            param_space=search_space,
        )
    tuner.fit()


if __name__ == '__main__':
    parser = get_model_dataset_arg_parser()
    parser.add_argument("--out", "-o", type=str, help="Optional out directory to write generated results to.")
    parser.add_argument("--search_alg", "-s", type=str, default="random", help="Search algorithm to use.")
    parser.add_argument("--cuda_ids", nargs="+", type=int, default=[0], help="Cuda device ids to use.")
    parser.add_argument("--gen_args", type=yaml.safe_load, default=None, help="Generation args.")
    parser.add_argument("--restore", "-r", action="store_true", help="Restore from checkpoint.")
    parser.set_defaults(restore=False)
    args = parser.parse_args()

    tune_inference(args)
