from neural_contact_fields.data.tool_dataset import ToolDataset
import neural_contact_fields.single_tool_neural_contact_field.config
from neural_contact_fields.pretrain_object_module.training import Trainer


def get_model(cfg, dataset: ToolDataset, device=None):
    model = neural_contact_fields.single_tool_neural_contact_field.config.get_model(cfg, dataset, device=device)
    return model


def get_trainer(model, optimizer, cfg, logger, vis_dir, device=None):
    trainer = Trainer(model, optimizer, logger, cfg["training"]["loss_weights"], vis_dir, device)

    return trainer
