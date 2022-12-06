from neural_contact_fields.data.tool_dataset import ToolDataset
from neural_contact_fields.single_tool_neural_contact_field.models.concat_mlp import \
    SingleToolNeuralContactField
from neural_contact_fields.single_tool_neural_contact_field.training import Trainer


def get_model(cfg, device=None):
    model_cfg = cfg["model"]

    model = SingleToolNeuralContactField(num_trials=dataset.num_trials, z=model_cfg["z"],
                                         forward_deformation=model_cfg["forward_deformation"], device=device)
    return model


def get_trainer(cfg, model, device=None):
    trainer = Trainer(cfg, model, device)
    return trainer
