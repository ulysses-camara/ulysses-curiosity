"""Base class for a probing task."""
import typing as t
import abc

import torch
import torch.nn


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


class DummyProbingTask(BaseProbingTask):
    """Dummy probing task used as placeholder."""

    def __init__(self, *args: t.Any, **kwargs: t.Any):
        # pylint: disable='unused-argument'
        super().__init__(
            probing_dataloader=torch.utils.data.DataLoader([0]),
            loss_fn=torch.nn.CrossEntropyLoss(),
        )
