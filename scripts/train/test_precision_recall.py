import argparse

import mmint_utils
import neural_contact_fields.metrics as metrics
import torch


def test_precision_recall(pred_dict: dict):
    # Dataset data.
    sdf = pred_dict["gt"]["sdf"]
    in_object = sdf < 0.0
    in_contact = pred_dict["gt"]["in_contact"] > 0.5

    # Prediction data.
    pred_sdf = pred_dict["pred"]["sdf"][0]
    pred_in_object = pred_sdf < 0.0
    pred_contact = pred_dict["pred"]["in_contact"][0] > 0.5

    in_object_pr = metrics.precision_recall(torch.from_numpy(pred_in_object), torch.from_numpy(in_object))
    in_contact_pr = metrics.precision_recall(torch.from_numpy(pred_contact), torch.from_numpy(in_contact))

    print("In Object PR:")
    print(in_object_pr)

    print("In Contact PR:")
    print(in_contact_pr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vis pred vs data.")
    parser.add_argument("pred", type=str)
    args = parser.parse_args()

    pred_dict_ = mmint_utils.load_gzip_pickle(args.pred)

    test_precision_recall(pred_dict_)
