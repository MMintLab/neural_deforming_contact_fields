import argparse
import pdb

from vedo import Plotter, Points, Arrows
import mmint_utils
import neural_contact_fields.utils.vedo_utils as vedo_utils


def vis_object_module_pretraining(pred_dict: dict):
    query_points = pred_dict["query_points"]
    pred_sdf = pred_dict["pred_sdf"]
    sdf = pred_dict["sdf"]
    normals = pred_dict["pred_normals"]

    plt = Plotter(shape=(2, 2))
    plt.at(0).show(Points(query_points), vedo_utils.draw_origin(), "All Sample Points")
    plt.at(1).show(Points(query_points[sdf <= 0.0], c="b"), vedo_utils.draw_origin(), "Occupied Points (GT)")
    plt.at(2).show(Points(query_points[pred_sdf <= 0.0], c="b"), vedo_utils.draw_origin(), "Occupied Points (Pred)")
    plt.at(3).show(Points(query_points[sdf == 0.0]),
                   Arrows(query_points[sdf == 0.0], query_points[sdf == 0.0] + (0.01 * normals)[sdf == 0.0]),
                   "Normals (Pred)")
    plt.interactive().close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vis object module pretraining.")
    parser.add_argument("data_fn", type=str, help="File with saved predictions.")
    args = parser.parse_args()

    pred_dict_ = mmint_utils.load_gzip_pickle(args.data_fn)

    vis_object_module_pretraining(pred_dict_)
