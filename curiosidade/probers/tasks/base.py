import abc

import torch


class BaseProbingTask(abc.ABC):
    def __init__(
        self,
        probing_dataloader: torch.utils.data.DataLoader,
        loss_fn,
        task_name: str = "unnamed_task",
    ):
        self.task_name = task_name
        self.probing_dataloader = probing_dataloader
        self.loss_fn = loss_fn
