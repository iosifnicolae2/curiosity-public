import torch


def push_to_tensor(tensor, x):
    return torch.cat((tensor[1:], x))