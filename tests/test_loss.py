import unittest
import torch

import neural_contact_fields.loss as loss


class TestLoss(unittest.TestCase):

    def test_sdf_loss(self):
        gt_sdf = torch.zeros([1, 1000]).float()
        pred_sdf = torch.ones([1, 1000]).float()

        sdf_loss = loss.sdf_loss(gt_sdf, pred_sdf)
        self.assertTrue(torch.allclose(sdf_loss, torch.tensor(1.0)))

        pred_sdf = torch.rand([1, 1000]).float()
        sdf_loss = loss.sdf_loss(gt_sdf, pred_sdf)
        self.assertTrue(torch.allclose(sdf_loss, torch.abs(pred_sdf).mean()))

        gt_sdf = torch.tensor([[0.1, -0.2, 0.0, 0.0, 0.4, -1.1]])
        pred_sdf = torch.tensor([[0.4, 0.0, 0.1, -0.2, 0.2, -0.9]])
        gt_sdf_loss = torch.tensor([0.3, 0.2, 0.1, 0.2, 0.2, 0.1]).mean()
        sdf_loss = loss.sdf_loss(gt_sdf, pred_sdf)
        self.assertTrue(torch.allclose(sdf_loss, gt_sdf_loss))


if __name__ == '__main__':
    unittest.main()
