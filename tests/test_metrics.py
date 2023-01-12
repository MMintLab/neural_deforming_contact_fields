import unittest

import numpy as np
import torch

import neural_contact_fields.metrics as ncf_metrics


class TestMetrics(unittest.TestCase):

    def test_binary_accuracy(self):
        gt = torch.from_numpy(np.array([True, False, False, True, True]))
        pred = torch.from_numpy(np.array([False, True, False, True, True]))

        binary_accuracy_gt = ncf_metrics.binary_accuracy(gt, gt)
        self.assertTrue(torch.allclose(binary_accuracy_gt, torch.tensor(1.0)))

        binary_accuracy = ncf_metrics.binary_accuracy(pred, gt)
        self.assertTrue(torch.allclose(binary_accuracy, torch.tensor(0.6)))

    def test_binary_accuracy_dim(self):
        gt = torch.from_numpy(np.array([[True, False, True], [False, False, True]]))
        pred = torch.from_numpy(np.array([[True, True, True], [True, True, True]]))

        binary_accuracy = ncf_metrics.binary_accuracy(gt, gt)
        self.assertTrue(torch.allclose(binary_accuracy, torch.tensor([1.0, 1.0])))

        binary_accuracy = ncf_metrics.binary_accuracy(pred, gt)
        self.assertTrue(torch.allclose(binary_accuracy, torch.tensor([2.0 / 3.0, 1.0 / 3.0])))

        pred = torch.from_numpy(np.array([[True, False, False], [False, False, True]]))
        binary_accuracy = ncf_metrics.binary_accuracy(pred, gt)
        self.assertTrue(torch.allclose(binary_accuracy, torch.tensor([2.0 / 3.0, 1.0])))

    def test_precision_recall(self):
        gt = torch.from_numpy(np.array([True, False, False, True, True]))
        pred = torch.from_numpy(np.array([False, True, False, True, True]))

        pr_dict = ncf_metrics.precision_recall(gt, gt)
        self.assertTrue((pr_dict["tp"] == torch.tensor([True, False, False, True, True])).all())
        self.assertTrue((pr_dict["tn"] == torch.tensor([False, True, True, False, False])).all())
        self.assertTrue((pr_dict["fp"] == torch.tensor([False, False, False, False, False])).all())
        self.assertTrue((pr_dict["fn"] == torch.tensor([False, False, False, False, False])).all())
        self.assertTrue(torch.allclose(pr_dict["precision"], torch.tensor(1.0)))
        self.assertTrue(torch.allclose(pr_dict["recall"], torch.tensor(1.0)))

        pr_dict = ncf_metrics.precision_recall(pred, gt)
        self.assertTrue((pr_dict["tp"] == torch.tensor([False, False, False, True, True])).all())
        self.assertTrue((pr_dict["tn"] == torch.tensor([False, False, True, False, False])).all())
        self.assertTrue((pr_dict["fp"] == torch.tensor([False, True, False, False, False])).all())
        self.assertTrue((pr_dict["fn"] == torch.tensor([True, False, False, False, False])).all())
        self.assertTrue(torch.allclose(pr_dict["precision"], torch.tensor(2.0 / 3.0)))
        self.assertTrue(torch.allclose(pr_dict["recall"], torch.tensor(2.0 / 3.0)))

        pred = torch.from_numpy(np.array([True, True, False, True, True]))
        pr_dict = ncf_metrics.precision_recall(pred, gt)
        self.assertTrue((pr_dict["tp"] == torch.tensor([True, False, False, True, True])).all())
        self.assertTrue((pr_dict["tn"] == torch.tensor([False, False, True, False, False])).all())
        self.assertTrue((pr_dict["fp"] == torch.tensor([False, True, False, False, False])).all())
        self.assertTrue((pr_dict["fn"] == torch.tensor([False, False, False, False, False])).all())
        self.assertTrue(torch.allclose(pr_dict["precision"], torch.tensor(3.0 / 4.0)))
        self.assertTrue(torch.allclose(pr_dict["recall"], torch.tensor(1.0)))


if __name__ == '__main__':
    unittest.main()
