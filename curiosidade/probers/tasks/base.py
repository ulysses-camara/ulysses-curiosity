"""Base class for a probing task."""
import abc

import torch


LossFunctionType = t.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class BaseProbingTask(abc.ABC):
    """Base class for a probing task.

    Parameters
    ----------
    probing_dataloader : torch.utils.data.DataLoader
        Train probing dataloader.

    loss_fn : t.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Loss function related to the probing task.

    task_name : str, default="unnamed_task"
        Probing task name.
    """

    def __init__(
        self,
        probing_dataloader: torch.utils.data.DataLoader,
        loss_fn: LossFunctionType,
        task_name: str = "unnamed_task",
    ):
        self.task_name = task_name
        self.probing_dataloader = probing_dataloader
        self.loss_fn = loss_fn
