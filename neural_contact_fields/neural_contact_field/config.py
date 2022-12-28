from neural_contact_fields.neural_contact_field.models.virdo_ncf import VirdoNCF
from neural_contact_fields.neural_contact_field.training import Trainer


def get_model(cfg, device=None):
    model_cfg = cfg["model"]

    model = VirdoNCF(model_cfg["z_object_size"], model_cfg["z_deform_size"], device)
    return model


def get_trainer(cfg, model, device=None):
    trainer = Trainer(cfg, model, device)
    return trainer
