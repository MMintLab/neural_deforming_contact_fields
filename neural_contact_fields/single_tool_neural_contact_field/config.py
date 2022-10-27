from neural_contact_fields.single_tool_neural_contact_field.models.single_tool_neural_contact_field import \
    SingleToolNeuralContactField
from neural_contact_fields.single_tool_neural_contact_field.training import Trainer


def get_model(cfg, device=None):
    model = SingleToolNeuralContactField(device)
    return model


def get_trainer(model, optimizer, cfg, logger, vis_dir, device=None):
    trainer = Trainer(model, optimizer, logger, None, vis_dir, device)

    return trainer
