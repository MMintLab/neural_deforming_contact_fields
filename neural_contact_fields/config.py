from neural_contact_fields.data.real_tool_dataset import RealToolDataset
from neural_contact_fields.data.tool_dataset import ToolDataset
from neural_contact_fields.data.pretrain_object_module_dataset import PretrainObjectModuleDataset
from torchvision import transforms
from neural_contact_fields import neural_contact_field
from neural_contact_fields import explicit_baseline

method_dict = {
    'neural_contact_field': neural_contact_field,
    'explicit_baseline': explicit_baseline,
}


def get_model(cfg, dataset, device=None):
    """
    Args:
    - cfg (dict): training config.
    - dataset (dataset): training dataset (in case model depends).
    - device (device): pytorch device.
    """
    method = cfg['method']
    print("cpt1")
    model = method_dict[method].config.get_model(cfg, dataset, device=device)
    print("cpt2")
    return model


def get_trainer(cfg, model, device=None):
    """
    Return trainer instance.

    Args:
    - cfg (dict): training config
    - model (nn.Module): model which is used
    - device (device): pytorch device
    """
    method = cfg['method']
    trainer = method_dict[method].config.get_trainer(cfg, model, device)
    return trainer


def get_generator(cfg, model, device=None):
    """
    Return generator instance.

    Args:
    - cfg (dict): configuration dict
    - model (nn.Module): model which is used
    - device (torch.device): pytorch device
    """
    method = cfg['method']
    generator = method_dict[method].config.get_generator(cfg, model, device)
    return generator


def get_visualizer(cfg, model, device=None, visualizer_args=None):
    """
    Return visualizer instance.

    Args:
    - cfg (dict): configuration dict
    - model (nn.Module): model which is used
    - device (torch.device): pytorch device
    """
    method = cfg['method']
    visualizer = method_dict[method].config.get_visualizer(cfg, model, device, visualizer_args)
    return visualizer


def get_dataset(mode, cfg):
    """
    Args:
    - mode (str): dataset mode [train, val, test].
    - cfg (dict): training config.
    """
    dataset_type = cfg['data'][mode]['dataset']

    # Build dataset transforms.
    transforms_ = get_transforms(cfg)

    if dataset_type == "ToolDataset":
        dataset = ToolDataset(cfg["data"][mode]["dataset_dir"], transform=transforms_)
    elif dataset_type == "PretrainObjectModuleDataset":
        dataset = PretrainObjectModuleDataset(cfg["data"][mode]["dataset_fn"], transform=transforms_)
    elif dataset_type == "RealToolDataset":
        dataset = RealToolDataset(cfg["data"][mode]["dataset_dir"], transform=transforms_)
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
