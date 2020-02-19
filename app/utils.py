from threading import Thread

import torch


def push_to_tensor(tensor, x):
    while len(x.shape) < len(tensor.shape):
        x = x.unsqueeze(0)

    return torch.cat((tensor, x), 1)[:, 1:, :]


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None, None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return
