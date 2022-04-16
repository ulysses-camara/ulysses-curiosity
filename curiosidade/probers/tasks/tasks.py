"""Probing task classes."""
import typing as t

import torch
import torch.nn

from . import base


class ProbingTaskSentenceLength(base.BaseProbingTask):
    """Preconfigured Sentence length (SentLen) probing task.

    Based on [1]_.

    References
    ----------
    .. [1] Alexis Conneau, German Kruszewski, Guillaume Lample, Loïc Barrault, and Marco Baroni.
       2018. What you can cram into a single $&!#* vector: Probing sentence embeddings for
       linguistic properties. In Proceedings of the 56th Annual Meeting of the Association for
       Computational Linguistics (Volume 1: Long Papers), pages 2126–2136, Melbourne, Australia.
       Association for Computational Linguistics.
    """

    def __init__(self, batch_size_train: int = 16, batch_size_eval: int = 32):
        dataset_uri_train = "todo"
        dataset_uri_eval = "todo"
        dataset_uri_test = "todo"

        super().__init__(
            loss_fn=torch.nn.CrossEntropyLoss(),
            output_dim=6,
            dataset_uri_or_dataloader_train=dataset_uri_train,
            dataset_uri_or_dataloader_eval=dataset_uri_eval,
            dataset_uri_or_dataloader_test=dataset_uri_test,
            task_type="classification",
            task_name="sentence length (sentlen)",
        )


class ProbingTaskWordContent(base.BaseProbingTask):
    pass


class ProbingTaskBigramShift(base.BaseProbingTask):
    pass


class ProbingTaskTreeDepth(base.BaseProbingTask):
    pass


class ProbingTaskTopConstituent(base.BaseProbingTask):
    pass


class ProbingTaskTense(base.BaseProbingTask):
    pass


class ProbingTaskSubjectNumber(base.BaseProbingTask):
    pass


class ProbingTaskObjectNumber(base.BaseProbingTask):
    pass


class ProbingTaskSOMO(base.BaseProbingTask):
    pass


class ProbingTaskCoordinationInversion(base.BaseProbingTask):
    pass


class ProbingTaskCustom(base.BaseProbingTask):
    """Custom probing task.

    Parameters
    ----------
    output_dim : int
        Dimension of the probing model final output. If the task type is classification, then this
        argument is usually the number of distinct labels present the probing dataset. If its type
        is regression task, it is usually 1.

    loss_fn : t.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Loss function related to the probing task.

    probing_dataloader_train : torch.utils.data.DataLoader
        Train probing dataloader.

    probing_dataloader_eval : torch.utils.data.DataLoader or None, default=None
        Evaluation probing dataloader.

    probing_dataloader_test : torch.utils.data.DataLoader or None, default=None
        Test probing dataloader.

    task_name : str, default="unnamed_task"
        Probing task name.

    task_type : {'classification', 'regression', 'mixed'}, default='classification'
        Type of task. Used only as reference, since it is the `loss_fn` that dictates
        how exactly the labels must be formatted.
    """

    def __init__(
        self,
        output_dim: int,
        loss_fn: base.LossFunctionType,
        probing_dataloader_train: torch.utils.data.DataLoader,
        probing_dataloader_eval: t.Optional[str] = None,
        probing_dataloader_test: t.Optional[str] = None,
        task_name: str = "unnamed_task",
        task_type: t.Literal["classification", "regression", "mixed"] = "classification",
    ):
        super().__init__(
            dataset_uri_or_dataloader_train=probing_dataloader_train,
            dataset_uri_or_dataloader_eval=probing_dataloader_eval,
            dataset_uri_or_dataloader_test=probing_dataloader_test,
            loss_fn=loss_fn,
            output_dim=output_dim,
            task_name=task_name,
            task_type=task_type,
        )
