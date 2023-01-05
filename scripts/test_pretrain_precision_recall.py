import argparse
import mmint_utils
import neural_contact_fields.metrics as metrics
import torch
from neural_contact_fields.utils import vedo_utils
from vedo import Plotter, Points


def test_pretrain_precision_recall(data_fn: str, vis: bool = False):
    pred_dict = mmint_utils.load_gzip_pickle(data_fn)

    pred_sdf = pred_dict["pred_sdf"] < 0.0
    sdf = pred_dict["sdf"] < 0.0

    in_object_pr = metrics.precision_recall(torch.from_numpy(pred_sdf),
                                            torch.from_numpy(sdf))

    print(in_object_pr)

    if vis:
        qp = pred_dict["query_points"]
        fp = in_object_pr["fp"]
        fn = in_object_pr["fn"]

        plt = Plotter(shape=(1, 3))
        plt.at(0).show(Points(qp[sdf]), vedo_utils.draw_origin(), "GT Points")
        plt.at(1).show(Points(qp[fp]), vedo_utils.draw_origin(), "False Positive Points")
        plt.at(2).show(Points(qp[fn]), vedo_utils.draw_origin(), "False Negative Points")
        plt.interactive().close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vis object module pretraining.")
    parser.add_argument("data_fn", type=str, help="File with saved predictions.")
    parser.add_argument("-v", "--vis", dest="vis", action="store_true", help="Visualize.")
    parser.set_defaults(vis=False)
    args = parser.parse_args()

    test_pretrain_precision_recall(args.data_fn, args.vis)
