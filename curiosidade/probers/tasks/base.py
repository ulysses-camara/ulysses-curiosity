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
    probing_dataloader_train : torch.utils.data.DataLoader
        Train probing dataloader.

    probing_dataloader_eval : torch.utils.data.DataLoader or None, default=None
        Evaluation probing dataloader.

    probing_dataloader_test : torch.utils.data.DataLoader or None, default=None
        Test probing dataloader.

    loss_fn : t.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Loss function related to the probing task.

    task_name : str, default="unnamed_task"
        Probing task name.
    """

    def __init__(
        self,
        loss_fn: LossFunctionType,
        probing_dataloader_train: torch.utils.data.DataLoader,
        probing_dataloader_eval: t.Optional[torch.utils.data.DataLoader] = None,
        probing_dataloader_test: t.Optional[torch.utils.data.DataLoader] = None,
        task_name: str = "unnamed_task",
    ):
        self.task_name = task_name
        self.loss_fn = loss_fn

        self.probing_dataloader_train = probing_dataloader_train
        self.probing_dataloader_eval = probing_dataloader_eval
        self.probing_dataloader_test = probing_dataloader_test

    @property
    def has_eval(self) -> bool:
        """Check whether task has evaluation dataset associated with it."""
        return self.probing_dataloader_eval is not None

    @property
    def has_test(self) -> bool:
        """Check whether task has test dataset associated with it."""
        return self.probing_dataloader_test is not None


class DummyProbingTask(BaseProbingTask):
    """Dummy probing task used as placeholder."""

    def __init__(self, *args: t.Any, **kwargs: t.Any):
        # pylint: disable='unused-argument'
        dummy_df = torch.utils.data.TensorDataset(torch.empty(0))
        dummy_dl = torch.utils.data.DataLoader(dummy_df)
        super().__init__(
            probing_dataloader_train=dummy_dl,
            loss_fn=torch.nn.CrossEntropyLoss(),
        )
