from neural_contact_fields.undcf.models.undcf import UNDCF
from neural_contact_fields.undcf.training import Trainer
from neural_contact_fields.undcf.generation import Generator
from neural_contact_fields.undcf.visualization import Visualizer


def get_model(cfg, dataset, device=None):
    model_cfg = cfg["model"]
    model_method = model_cfg["method"]

    if model_method == "neural_contact_field":
        model = UNDCF(model_cfg["z_deform_size"], model_cfg["z_wrench_size"], device)
    else:
        raise Exception("Unknown model method: %s" % model_method)
    return model


def get_trainer(cfg, model, device=None):
    trainer = Trainer(cfg, model, device)
    return trainer


def get_generator(cfg, model, generation_cfg, device=None):
    generator = Generator(cfg, model, generation_cfg, device)
    return generator


def get_visualizer(cfg, model, device=None, visualizer_args=None):
    visualizer = Visualizer(cfg, model, device, visualizer_args)
    return visualizer
