import typing as t
import abc

import torch
import torch.nn


class BaseProber(abc.ABC):
    def __init__(
        self, probing_model: torch.nn.Module, task, optim_fn: t.Type[torch.optim.Optimizer]
    ):
        self.probing_model = probing_model
        self.task = task
        self.optim = optim_fn(self.probing_model.parameters())
