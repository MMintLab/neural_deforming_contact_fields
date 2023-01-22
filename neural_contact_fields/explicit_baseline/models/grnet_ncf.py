import neural_contact_fields.explicit_baseline.grnet.utils as utils
import neural_contact_fields.explicit_baseline.grnet.utils.helpers

import torch
from neural_contact_fields.explicit_baseline.models.grnet_modules import GRNet_encoder, GRNet_decoder
from neural_contact_fields.neural_contact_field.models.neural_contact_field import NeuralContactField


from neural_contact_fields.explicit_baseline.grnet.config import cfg
from neural_contact_fields.models import mlp

class Grnet(NeuralContactField):
    def __init__(self, num_objects: int, num_trials: int, z_object_size: int, z_deform_size: int, z_wrench_size: int,
                 device=None):
        super().__init__(z_object_size, z_deform_size, z_wrench_size)
        self.device = device

        # wrench encoder.
        self.wrench_encoder = mlp.build_mlp(6, self.z_wrench_size, hidden_sizes=[16], device=device)

        # multi-modal encoder.
        self.grnet_encoder = GRNet_encoder(cfg)
        self.grnet_encoder.apply(utils.helpers.init_weights)
        self.grnet_encoder.to(self.device)


        # deformed pcd decoder.
        self.deform_decoder = GRNet_decoder(cfg, z_deform_size = self.z_deform_size)
        self.deform_decoder.apply(utils.helpers.init_weights)
        self.deform_decoder.to(self.device)


        # contact patch decoder.
        self.contact_decoder = GRNet_decoder(cfg, z_deform_size = self.z_deform_size)
        self.contact_decoder.apply(utils.helpers.init_weights)
        self.contact_decoder.to(self.device)


    def encode_object(self, object_idx: torch.Tensor):
        return self.object_code(object_idx)


    def load_pretrained_model(self, pretrain_file: str, load_pretrain_cfg: dict):
        pass

    def encode_trial(self, object_idx: torch.Tensor, trial_idx: torch.Tensor):
        z_object = self.encode_object(object_idx)
        z_trial = self.trial_code(trial_idx)
        return z_object, z_trial

    def encode_wrench(self, wrench: torch.Tensor):
        z_wrench = self.wrench_encoder(wrench)
        return z_wrench

    def forward(self, query_points: torch.Tensor, z_wrench: torch.Tensor):

        z_wrench = z_wrench.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        pcd_feat = self.grnet_encoder(query_points)
        sparse_df_ptcloud, dense_df_ptcloud = self.deform_decoder(query_points, z_wrench, pcd_feat)
        sparse_ct_ptcloud, dense_ct_ptcloud = self.contact_decoder(query_points, z_wrench,  pcd_feat)


        out_dict = {'sparse_df_cloud': sparse_df_ptcloud,
                    'dense_df_ptcloud': dense_df_ptcloud,
                    'sparse_ct_ptcloud':sparse_ct_ptcloud,
                    'dense_ct_ptcloud':dense_ct_ptcloud,
                    }
        return out_dict



    #
    # def regularization_loss(self, out_dict: dict):
    #     def_hypo_params = out_dict["def_hypo_params"]
    #     def_hypo_loss = ncf_losses.hypo_weight_loss(def_hypo_params)
    #     sdf_hypo_params = out_dict["sdf_hypo_params"]
    #     sdf_hypo_loss = ncf_losses.hypo_weight_loss(sdf_hypo_params)
    #     in_contact_hypo_params = out_dict["in_contact_hypo_params"]
    #     in_contact_hypo_loss = ncf_losses.hypo_weight_loss(in_contact_hypo_params)
    #     return def_hypo_loss + sdf_hypo_loss + in_contact_hypo_loss
