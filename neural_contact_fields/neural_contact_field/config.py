from neural_contact_fields.neural_contact_field.models.virdo_ncf import VirdoNCF
from neural_contact_fields.neural_contact_field.training import Trainer
from neural_contact_fields.neural_contact_field.generation import Generator


def get_model(cfg, dataset, device=None):
    model_cfg = cfg["model"]

    try:
        num_objects = dataset.num_objects
        num_trials = dataset.num_trials
    except:
        raise Exception("Training with unexpected dataset type: %s." % str(type(dataset)))

    model = VirdoNCF(num_objects, num_trials, model_cfg["z_object_size"], model_cfg["z_deform_size"],
                     model_cfg["z_wrench_size"], device)
    return model


def get_trainer(cfg, model, device=None):
    trainer = Trainer(cfg, model, device)
    return trainer


def get_generator(cfg, model, device=None):
    generator = Generator(cfg, model, device)
    return generator
