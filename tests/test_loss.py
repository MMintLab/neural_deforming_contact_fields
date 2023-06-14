import unittest
from collections import OrderedDict

import numpy as np
import torch
from torch.distributions import Normal

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

    def test_normals_loss(self):
        gt_sdf = torch.tensor([0.0, 0.0, 0.0])
        gt_normals = torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])

        norm_loss = loss.surface_normal_loss(gt_sdf, gt_normals, gt_normals)
        self.assertTrue(torch.allclose(norm_loss, torch.tensor(0.0)))

        pred_normals = torch.tensor([[0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        norm_loss = loss.surface_normal_loss(gt_sdf, gt_normals, pred_normals)
        self.assertTrue(torch.allclose(norm_loss, torch.tensor(1.0)))

        pred_normals = torch.tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        norm_loss = loss.surface_normal_loss(gt_sdf, gt_normals, pred_normals)
        self.assertTrue(torch.allclose(norm_loss, torch.tensor(2.0)))

        gt_sdf = torch.tensor([0.0, -0.1, 0.1])
        pred_normals = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        norm_loss = loss.surface_normal_loss(gt_sdf, gt_normals, pred_normals)
        self.assertTrue(torch.allclose(norm_loss, torch.tensor(0.0)))

        gt_sdf = torch.tensor([0.0, 0.0])
        gt_normals = torch.tensor([[0.69631062, 0.69631062, 0.17407766], [0.37139068, 0.74278135, -0.55708601]])
        pred_normals = torch.tensor([[0.4, 0.4, 0.1], [-0.2, -0.4, 0.3]])
        norm_loss = loss.surface_normal_loss(gt_sdf, gt_normals, pred_normals)
        self.assertTrue(torch.allclose(norm_loss, torch.tensor(1.0)))

    def test_l2_loss(self):
        a = torch.tensor([1.0, 2.0, 3.0, 1.0])

        a_l2 = loss.l2_loss(a)
        a_l2_squared = loss.l2_loss(a, squared=True)
        self.assertTrue(torch.allclose(a_l2, torch.tensor(np.sqrt(15.0)).float()))
        self.assertTrue(torch.allclose(a_l2_squared, torch.tensor(15.0)))

        b = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        b_l2 = loss.l2_loss(b)
        b_l2_squared = loss.l2_loss(b, squared=True)
        self.assertTrue(torch.allclose(b_l2, torch.tensor(1.36603)))
        self.assertTrue(torch.allclose(b_l2_squared, torch.tensor(2.0)))

    def test_surface_chamfer_loss(self):
        pc_1 = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]]).unsqueeze(0).float()
        sdf_1 = torch.zeros(3).unsqueeze(0).float()
        pc_2 = torch.tensor([[1, 0, 0], [2, 0, 0], [0, 2, 0.5], [0, 0, -0.5]]).unsqueeze(0).float()
        sdf_2 = torch.zeros(4).unsqueeze(0).float()

        chamfer_gt = ((0.25 + 0.0 + 1.25) / 3.0) + ((0.0 + 1.0 + 1.25 + 0.25) / 4.0)
        chamfer_dist = loss.surface_chamfer_loss(pc_1, sdf_1, sdf_2, pc_2)
        self.assertTrue(torch.allclose(chamfer_dist, torch.tensor(chamfer_gt)))

    def test_hypo_weight_loss(self):
        hypo_weights = OrderedDict()
        hypo_weights["l1"] = torch.tensor([0.1, 1.0, 3.0, 0.4, -0.2])
        hypo_weights["l2"] = torch.tensor([-1.0, -0.3])

        hypo_w_loss = loss.hypo_weight_loss(hypo_weights)
        self.assertTrue(torch.allclose(hypo_w_loss, torch.tensor(1.614285714285714).float()))

    def test_heteroscedastic_bce_loss(self):
        means = torch.tensor([0.0, 3.0, -12.0, 0.0, 0.3]).float()
        variances = torch.tensor([1.0, 4.0, 1.0, 10.0, 0.1]).float()
        dist = Normal(means, variances)

        labels = torch.tensor([0, 1, 0, 1, 1]).float()

        bce_loss = loss.heteroscedastic_bce(dist, labels, n=50)

        self.assertTrue(bce_loss.shape == torch.Size([5]))


if __name__ == '__main__':
    unittest.main()
