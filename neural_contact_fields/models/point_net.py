from pointnet.model import PointNetfeat
import torch.nn as nn


class PointNet(nn.Module):

    def __init__(self, z_dim, pooling):
        super().__init__()
        self.z_dim = z_dim
        self.pooling = pooling
        self.point_net_feat = PointNetfeat(self.z_dim, pooling=self.pooling)

    def forward(self, x):
        if len(x.shape) == 4:
            original_shape = x.shape
            x_in = x.reshape(-1, x.shape[2], x.shape[3])
        else:
            x_in = x

        x_in = x_in.transpose(1, 2)  # Transpose last two dimensions.
        z_v, _, _ = self.point_net_feat(x_in)

        if len(x.shape) == 4:
            z_v = z_v.reshape(original_shape[0], original_shape[1], -1)

        return z_v
