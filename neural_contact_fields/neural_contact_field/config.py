from neural_contact_fields.data.pretrain_object_module_dataset import PretrainObjectModuleDataset
from neural_contact_fields.data.tool_dataset import ToolDataset
from neural_contact_fields.neural_contact_field.models.virdo_ncf import VirdoNCF
from neural_contact_fields.neural_contact_field.training import Trainer


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
