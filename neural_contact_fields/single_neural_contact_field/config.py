from neural_contact_fields.single_neural_contact_field.models.single_neural_contact_field import \
    SingleNeuralContactField
from neural_contact_fields.single_neural_contact_field.training import Trainer


def get_model(cfg, device=None):
    model = SingleNeuralContactField(device)
    return model


def get_trainer(model, optimizer, cfg, logger, vis_dir, device=None):
    trainer = Trainer(model, optimizer, logger, None, vis_dir, device)

    return trainer
