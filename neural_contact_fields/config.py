import os

from neural_contact_fields.data.single_tool_dataset import SingleToolDataset
from neural_contact_fields.data.tool_dataset import ToolDataset
from torchvision import transforms
from neural_contact_fields import single_neural_contact_field

method_dict = {
    'single_neural_contact_field': single_neural_contact_field,
}


def get_model(cfg, device=None):
    """
    Args:
    - cfg (dict): training config.
    - device (device): pytorch device.
    """
    method = cfg['method']
    model = method_dict[method].config.get_model(
        cfg, device=device)
    return model


def get_trainer(model, optimizer, cfg, logger, vis_dir, device=None):
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
    trainer = method_dict[method].config.get_trainer(
        model, optimizer, cfg, logger, vis_dir, device)
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
