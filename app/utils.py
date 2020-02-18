import torch


def push_to_tensor(tensor, x):
    while len(x.shape) < len(tensor.shape):
        x = x.unsqueeze(0)

    return torch.cat((tensor, x), 1)[:, 1:, :]
