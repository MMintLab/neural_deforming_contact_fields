from tqdm import trange
import os

import mmint_utils
from neural_contact_fields import config
from neural_contact_fields.utils.args_utils import get_model_dataset_arg_parser, load_model_dataset_from_args
from neural_contact_fields.utils.model_utils import load_generation_cfg
from neural_contact_fields.utils.results_utils import load_gt_results, metrics_to_statistics, write_results

from ray.air import session
from ray import tune, air

from scripts.eval_results import eval_example
from scripts.generate import generate_example


def tune_inference(args):
    out_dir = args.out

    # Set visible gpus to the one provided.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id)
    args.cuda_id = 0  # Change index to be into visible devices list.

    # Make out dir.
    mmint_utils.make_dir(out_dir)

    # Define gen/eval function use by RayTune.
    def eval_hyper_params(param_config: dict):
        model_cfg, model, dataset, device = load_model_dataset_from_args(args)
        model.eval()

        # Load generate cfg, if present.
        base_generation_cfg = load_generation_cfg(model_cfg, args.model_file)

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

        # Compute stats on metrics.
        metrics_stats_dict = metrics_to_statistics(metrics_dicts)
        session.report(metrics_stats_dict)

    # Define search space.
    search_space = {
        "contact_threshold": tune.quniform(0.0, 1.0, 0.1),
        "embed_weight": tune.loguniform(1e-6, 1.0),
        "iter_limit": tune.qrandint(100, 1000, 100),
    }

    # Setup RayTune.
    trainable_with_resources = tune.with_resources(eval_hyper_params, {"cpu": 16, "gpu": 1})
    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(metric="patch_chamfer_distance_mean", mode="min", num_samples=10),
        run_config=air.RunConfig(local_dir=out_dir, name="tune_inference"),
        param_space=search_space,
    )
    tuner.fit()


if __name__ == '__main__':
    parser = get_model_dataset_arg_parser()
    parser.add_argument("--out", "-o", type=str, help="Optional out directory to write generated results to.")
    args = parser.parse_args()

    tune_inference(args)
