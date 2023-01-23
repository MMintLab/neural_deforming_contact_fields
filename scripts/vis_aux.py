import yaml
from neural_contact_fields import config
from neural_contact_fields.utils.args_utils import get_model_dataset_arg_parser, load_model_dataset_from_args
from tqdm import trange


def vis_aux(model_cfg, model, dataset, device, mode, vis_args):
    model.eval()
    vis_pretrain = mode == "pretrain"

    # Load visualizer.
    visualizer = config.get_visualizer(model_cfg, model, device, vis_args)

    # Go through dataset and generate!
    for idx in trange(len(dataset)):
        data_dict = dataset[idx]

        if vis_pretrain:
            visualizer.visualize_pretrain(data_dict)
        else:
            visualizer.visualize_train(data_dict)


if __name__ == '__main__':
    parser = get_model_dataset_arg_parser()
    parser.add_argument("--vis_args", type=yaml.safe_load, default=None, help="Visualization args.")
    args = parser.parse_args()

    model_cfg_, model_, dataset_, device_ = load_model_dataset_from_args(args)
    vis_aux(model_cfg_, model_, dataset_, device_, mode=args.mode, vis_args=args.vis_args)
