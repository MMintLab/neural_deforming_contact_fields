from neural_contact_fields.data.tool_dataset import ToolDataset
from neural_contact_fields.single_tool_neural_contact_field.models.single_tool_neural_contact_field import \
    SingleToolNeuralContactField
from neural_contact_fields.single_tool_neural_contact_field.training import Trainer


def get_model(cfg, dataset: ToolDataset, device=None):
    model_cfg = cfg["model"]

    model = SingleToolNeuralContactField(num_trials=dataset.num_trials, z=model_cfg["z"],
                                         forward_deformation=model_cfg["forward_deformation"], device=device)
    return model


def get_trainer(model, optimizer, cfg, logger, vis_dir, device=None):
    trainer = Trainer(model, optimizer, logger, cfg["training"]["loss_weights"], vis_dir, device)

    return trainer
