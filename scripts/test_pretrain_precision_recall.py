import argparse
import mmint_utils
import neural_contact_fields.metrics as metrics
import torch


def test_pretrain_precision_recall(data_fn: str):
    pred_dict = mmint_utils.load_gzip_pickle(data_fn)

    pred_sdf = pred_dict["pred_sdf"] < 0.0
    sdf = pred_dict["sdf"] < 0.0

    in_object_pr = metrics.precision_recall(torch.from_numpy(pred_sdf),
                                            torch.from_numpy(sdf))

    print(in_object_pr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vis object module pretraining.")
    parser.add_argument("data_fn", type=str, help="File with saved predictions.")
    args = parser.parse_args()

    test_pretrain_precision_recall(args.data_fn)
