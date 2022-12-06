import os

from neural_contact_fields.data.tool_dataset import ToolDataset
from neural_contact_fields.data.pretrain_object_module_dataset import PretrainObjectModuleDataset
from torchvision import transforms
from neural_contact_fields import single_neural_contact_field
from neural_contact_fields import single_tool_neural_contact_field
from neural_contact_fields import pretrain_object_module

method_dict = {
    'single_neural_contact_field': single_neural_contact_field,
    'single_tool_neural_contact_field': single_tool_neural_contact_field,
    'pretrain_object_module': pretrain_object_module,
}


def get_model(cfg, device=None):
    """
    Args:
    - cfg (dict): training config.
    - dataset (dataset): training dataset
    - device (device): pytorch device.
    """
    method = cfg['method']
    model = method_dict[method].config.get_model(cfg, device=device)
    return model


def get_trainer(cfg, model, device=None):
    """
    Return trainer instance.

    Args:
    - model (nn.Module): model which is used
    - optimizer (optimizer): pytorch optimizer
    - cfg (dict): training config
    - logger (tensorboardX.SummaryWriter): logger for tensorboard
    - vis_dir (str): vis directory
    - device (device): pytorch device
    """
    method = cfg['method']
    trainer = method_dict[method].config.get_trainer(cfg, model, device)
    return trainer


def get_dataset(mode, cfg):
    """
    Args:
    - mode (str): dataset mode [train, val, test].
    - cfg (dict): training config.
    """
    dataset_type = cfg['data']['dataset']

    # Build dataset transforms.
    transforms_ = get_transforms(cfg)

    if dataset_type == "SingleToolDataset":
        dataset = SingleToolDataset(cfg["data"]["dataset_fn"], cfg["data"]["split"], cfg["data"]["tool_idx"],
                                    cfg["data"]["deformation_idx"], transform=transforms_)
    elif dataset_type == "ToolDataset":
        dataset = ToolDataset(cfg["data"]["dataset_dir"], transform=transforms_)
    elif dataset_type == "PretrainObjectModuleDataset":
        dataset = PretrainObjectModuleDataset(cfg["data"]["dataset_fn"], transform=transforms_)
    else:
        raise Exception("Unknown requested dataset type: %s" % dataset_type)

    return dataset


def get_transforms(cfg):
    transforms_info = cfg['data'].get('transforms')
    if transforms_info is None:
        return None

    transform_list = []
    for transform_info in transforms_info:
        transform_type = transform_info["type"]
        transform = None

        raise Exception("Unknown transform type: %s" % transform_type)

        transform_list.append(transform)

    composed = transforms.Compose(transform_list)
    return composed
