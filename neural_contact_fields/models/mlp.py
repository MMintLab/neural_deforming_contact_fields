import torch.nn as nn


def build_mlp_dict(mlp_dict: dict, device=None):
    return build_mlp(mlp_dict["input_size"], mlp_dict["output_size"], mlp_dict["hidden_sizes"], device)


def build_mlp(input_size, output_size, hidden_sizes, device=None):
    return MLP(input_size, output_size, hidden_sizes).to(device)


class MLP(nn.Module):
    """
    Model + code based on: https://github.com/wilson1yan/contrastive-forward-model/blob/master/cfm/models.py
    """

    def __init__(self, input_size, output_size, hidden_sizes=None):
        """
        Args:
        - input_size (int): input dim
        - output_size (int): output dim
        - hidden_sizes (list): list of ints with hidden dim sizes
        """
        super().__init__()
        # TODO: Parameterize activation function?

        if hidden_sizes is None:
            hidden_sizes = []
        model = []
        prev_h = input_size
        for h in hidden_sizes + [output_size]:
            model.append(nn.Linear(prev_h, h))
            model.append(nn.ReLU())
            prev_h = h
        model.pop()  # Pop last ReLU
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
