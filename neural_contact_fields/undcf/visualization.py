import numpy as np
import torch
import torch.nn as nn
from neural_contact_fields.utils import vedo_utils
from neural_contact_fields.utils.infer_utils import inference_by_optimization

from neural_contact_fields.visualization import BaseVisualizer
from vedo import Plotter, Points, Arrows


class Visualizer(BaseVisualizer):

    def __init__(self, cfg: dict, model: nn.Module, device: torch.device = None, visualizer_args: dict = None):
        super().__init__(cfg, model, device, visualizer_args)
        self.train_infer_latent = self.visualizer_args.get("train_infer_latent", True)
        self.embed_weight = self.visualizer_args.get("embed_weight", 0.0)
        self.def_weight = self.visualizer_args.get("def_weight", 0.0)

    def visualize_pretrain(self, data: dict):
        """
        Ensure pretrained object module is well trained.
        """
        object_idx = torch.from_numpy(data["object_idx"]).to(self.device)
        query_points = torch.from_numpy(data["query_point"]).float().to(self.device).unsqueeze(0)

        z_object = self.model.encode_object(object_idx)
        out_dict = self.model.forward_object_module(query_points, z_object)

        # Pull out what we want to visualize.
        query_points = query_points.squeeze(0).detach().cpu().numpy()
        sdf = data["sdf"]
        pred_sdf = out_dict["sdf"].squeeze(0).detach().cpu().numpy()
        pred_normals = out_dict["normals"].squeeze(0).detach().cpu().numpy()

        plt = Plotter(shape=(2, 2))
        plt.at(0).show(Points(query_points), vedo_utils.draw_origin(), "All Sample Points")
        plt.at(1).show(Points(query_points[sdf <= 0.0], c="b"), vedo_utils.draw_origin(), "Occupied Points (GT)")
        plt.at(2).show(Points(query_points[pred_sdf <= 0.0], c="b"), vedo_utils.draw_origin(), "Occupied Points (Pred)")
        plt.at(3).show(Points(query_points[sdf == 0.0]), vedo_utils.draw_origin(),
                       Arrows(query_points[sdf == 0.0], query_points[sdf == 0.0] + (0.01 * pred_normals)[sdf == 0.0]),
                       "Normals (Pred)")
        plt.interactive().close()

    def visualize_results(self, data_dict: dict, pred_dict: dict):
        # Dataset data.
        all_points = data_dict["query_point"]
        sdf = data_dict["sdf"]
        in_contact = data_dict["in_contact"] > 0.5

        # Prediction data.
        pred_sdf = pred_dict["sdf"][0].detach().cpu().numpy()
        pred_contact = pred_dict["in_contact"][0].detach().cpu().numpy() > 0.5
        pred_def = pred_dict["deform"][0].detach().cpu().numpy()
        pred_normals = pred_dict["normals"][0].detach().cpu().numpy()

        plt = Plotter(shape=(2, 3))
        plt.at(0).show(Points(all_points[sdf <= 0.0], c="b"), Points(all_points[in_contact], c="r"),
                       vedo_utils.draw_origin(), "Ground Truth")
        plt.at(1).show(Points(all_points[pred_sdf <= 0.0], c="b"),
                       vedo_utils.draw_origin(), "Predicted Surface")
        plt.at(2).show(Arrows(all_points, all_points - pred_def), "Predicted Deformations")
        plt.at(3).show(Points(all_points[pred_sdf <= 0.0], c="b"),
                       Points(all_points[np.logical_and(pred_sdf <= 0.0, pred_contact)], c="r"),
                       vedo_utils.draw_origin(), "Predicted Contact")
        plt.at(4).show(Points(all_points[pred_sdf <= 0.0], c="b"),
                       Points(all_points[pred_contact], c="r"), vedo_utils.draw_origin(),
                       "Predicted Contact (All)")
        plt.at(5).show(Points(all_points[sdf == 0.0]),
                       Arrows(all_points[sdf == 0.0], all_points[sdf == 0.0] + (0.01 * pred_normals)[sdf == 0.0]),
                       "Predicted Normals")
        plt.interactive().close()

    def visualize_train(self, data: dict):
        object_index = data["object_idx"]
        trial_index = data["trial_idx"]
        wrist_wrench = torch.from_numpy(data["wrist_wrench"]).to(self.device).float().unsqueeze(0)

        # Encode object idx/trial idx.
        if not self.train_infer_latent:
            z_object, z_trial = self.model.encode_trial(torch.from_numpy(object_index).to(self.device),
                                                        torch.from_numpy(trial_index).to(self.device))
        else:
            z_object = self.model.encode_object(torch.from_numpy(object_index).to(self.device))
            z_deform_, _ = inference_by_optimization(self.model,
                                                     get_surface_loss_fn(self.embed_weight, self.def_weight),
                                                     self.model.z_deform_size, 1, data,
                                                     device=self.device, verbose=True)
            z_trial = z_deform_.weight
        z_wrench = self.model.encode_wrench(wrist_wrench)

        # Get query points to sample.
        query_points = torch.from_numpy(data["query_point"]).to(self.device).float()
        pred_dict = self.model.forward(query_points.unsqueeze(0), z_trial, z_object, z_wrench)

        self.visualize_results(data, pred_dict)

    def visualize_test(self, data: dict):
        object_index = data["object_idx"]
        wrist_wrench = torch.from_numpy(data["wrist_wrench"]).to(self.device).float().unsqueeze(0)

        # Encode object idx/trial idx.
        z_object = self.model.encode_object(torch.from_numpy(object_index).to(self.device))
        z_deform_, _ = inference_by_optimization(self.model, get_surface_loss_fn(self.embed_weight, self.def_weight),
                                                 self.model.z_deform_size, 1, data,
                                                 device=self.device, verbose=True)
        z_trial = z_deform_.weight
        z_wrench = self.model.encode_wrench(wrist_wrench)

        # Get query points to sample.
        query_points = torch.from_numpy(data["query_point"]).to(self.device).float()
        pred_dict = self.model.forward(query_points.unsqueeze(0), z_trial, z_object, z_wrench)

        self.visualize_results(data, pred_dict)
