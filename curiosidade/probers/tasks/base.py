"""Base class for a probing task."""
import typing as t
import pathlib
import abc
import os
import json

import torch
import torch.nn
import pandas as pd

try:
    from typing_extensions import TypeAlias

except ImportError:
    from typing import TypeAlias  # type: ignore


LossFunctionType = t.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
ValidationFunctionType = t.Callable[[torch.Tensor, torch.Tensor], dict[str, float]]
DataLoaderType: TypeAlias = torch.utils.data.DataLoader[tuple[torch.Tensor, ...]]
DataLoaderGenericType = t.Union[str, pathlib.Path, DataLoaderType]


class BaseProbingTask(abc.ABC):
    """Base class for a probing task.

    Parameters
    ----------
    output_dim : int
        Dimension of the probing model final output. If the task type is classification, then this
        argument is usually the number of distinct labels present the probing dataset. If its type
        is regression task, it is usually 1.

    loss_fn : t.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Loss function related to the probing task.

    dataset_uri_or_dataloader_train : str or pathlib.Path or torch.utils.data.DataLoader
        Train probing dataset URI or DataLoader.

    dataset_uri_or_dataloader_eval : str or pathlib.Path or torch.utils.data.DataLoader or None,\
            default=None
        Optional evaluation probing dataset URI or DataLoader.

    dataset_uri_or_dataloader_test : str or pathlib.Path or torch.utils.data.DataLoader or None,\
            default=None
        Optional test probing dataset URI or DataLoader.

    metrics_fn : t.Callable[[torch.Tensor, torch.Tensor], dict[str, float]] or None,\
            default=None
        Validation function to compute extra scores from training, validation and test batches.
        As the first argument, it must receive a logit tensor of shape (batch_size, output_dim),
        and  a ground-truth label tensor os shape (batch_size,) as the second argument.
        The return value must always be a dictionary (or any other valid mapping) mapping the
        metric name and its computed value.
        If None, no extra validation metrics will be computed, and only the loss values will
        be returned as result.

    task_name : str, default="unnamed_task"
        Probing task name.

    task_type : {'classification', 'regression', 'mixed'}, default='classification'
        Type of task. Used only as reference, since it is the `loss_fn` that dictates
        how exactly the labels must be formatted.
    """

    VALID_DATA_DOMAINS: t.Final[frozenset[str, ...]] = frozenset(("general-pt-br",))

    def __init__(
        self,
        output_dim: int,
        loss_fn: LossFunctionType,
        labels_uri_or_map: t.Union[dict[str, int], t.Sequence[str], str],
        dataset_uri_or_dataloader_train: DataLoaderGenericType,
        dataset_uri_or_dataloader_eval: t.Optional[DataLoaderGenericType] = None,
        dataset_uri_or_dataloader_test: t.Optional[DataLoaderGenericType] = None,
        metrics_fn: t.Optional[ValidationFunctionType] = None,
        task_name: str = "unnamed_task",
        task_type: t.Literal["classification", "regression", "mixed"] = "classification",
        batch_size_train: t.Optional[int] = None,
        batch_size_eval: t.Optional[int] = None,
    ):
        if task_type not in {"classification", "regression", "mixed"}:
            raise ValueError(
                f"Provided 'task_type' unsupported ('{task_type}'). Must be either "
                "'classification', 'regression' or 'mixed'."
            )

        output_dim = int(output_dim)

        if output_dim <= 0:
            raise ValueError(f"Invalid 'output_dim' ({output_dim}), must be >= 1.")

        self.task_name = task_name
        self.metrics_fn = metrics_fn
        self.loss_fn = loss_fn
        self.task_type = task_type
        self.output_dim = output_dim

        dl_train: DataLoaderType
        dl_eval: t.Optional[DataLoaderType]
        dl_test: t.Optional[DataLoaderType]
        self.labels: t.Dict[str, int]

        if isinstance(labels_uri_or_map, dict):
            self.labels = labels_uri_or_map.copy()

        elif isinstance(labels_uri_or_map, (str, pathlib.Path)):
            with open(labels_uri_or_map, "r", encoding="utf-8") as f_in:
                labels = json.load(f_in)["sentence_length"]

            self.labels = {cls: ind for ind, cls in enumerate(labels)}

        else:
            self.labels = {cls: ind for ind, cls in enumerate(labels_uri_or_map)}

        if isinstance(dataset_uri_or_dataloader_train, (str, pathlib.Path)):
            dl_train = torch.utils.data.DataLoader(
                self._load_dataset(dataset_uri_or_dataloader_train, split="train"),
                batch_size=batch_size_train or 16,
                shuffle=True,
            )

        else:
            dl_train = dataset_uri_or_dataloader_train

        if isinstance(dataset_uri_or_dataloader_eval, (str, pathlib.Path)):
            dl_eval = torch.utils.data.DataLoader(
                self._load_dataset(dataset_uri_or_dataloader_eval, split="eval"),
                batch_size=batch_size_eval or 32,
                shuffle=False,
            )

        else:
            dl_eval = dataset_uri_or_dataloader_eval

        if isinstance(dataset_uri_or_dataloader_test, (str, pathlib.Path)):
            dl_test = torch.utils.data.DataLoader(
                self._load_dataset(dataset_uri_or_dataloader_test, split="test"),
                batch_size=batch_size_eval,
                shuffle=False,
            )

        else:
            dl_test = dataset_uri_or_dataloader_test

        self.probing_dataloader_train = dl_train
        self.probing_dataloader_eval = dl_eval
        self.probing_dataloader_test = dl_test

    @classmethod
    def check_if_domain_is_valid(cls, data_domain: str) -> None:
        """Check whether a given `data_domain` is currently supported."""
        if data_domain in cls.VALID_DATA_DOMAINS:
            return

        raise ValueError(f"Invalid '{data_domain=}'. Must be in {cls.VALID_DATA_DOMAINS}.")

    @property
    def has_eval(self) -> bool:
        """Check whether task has evaluation dataset associated with it."""
        return self.probing_dataloader_eval is not None

    @property
    def has_test(self) -> bool:
        """Check whether task has test dataset associated with it."""
        return self.probing_dataloader_test is not None

    @property
    def has_metrics(self) -> bool:
        """Check whether task has metric functions."""
        return self.metrics_fn is not None

    @property
    def num_classes(self) -> int:
        """Return number of classes of the probing task."""
        return len(self.labels)

    def _load_dataset(
        self,
        dataset_split_uri: t.Union[pathlib.Path, str],
        split: t.Literal["train", "eval", "test"],
    ) -> torch.utils.data.Dataset[tuple[torch.Tensor, ...]]:
        """Load a prepared dataset."""
        df_split = pd.read_csv(
            dataset_split_uri,
            sep="\t",
            header=0,
            usecols=["label", "content"],
            dtype=str,
        )

        labels = df_split["label"].map(self.labels)

        content = df_split["content"].tolist()
        labels = labels.tolist()

        try:
            out = self.fn_text_to_tensor(content, labels, split=split)

        except AttributeError:
            out = self.fn_text_to_tensor(content, labels)

        return out


class DummyProbingTask(BaseProbingTask):
    """Dummy probing task used as placeholder."""

    def __init__(self, *args: t.Any, **kwargs: t.Any):
        # pylint: disable='unused-argument'
        dummy_df = torch.utils.data.TensorDataset(torch.empty(0), torch.empty(0))
        dummy_dl = torch.utils.data.DataLoader(dummy_df)
        super().__init__(
            dataset_uri_or_dataloader_train=dummy_dl,
            loss_fn=torch.nn.CrossEntropyLoss(),
            labels_uri_or_map=["A", "B"],
            output_dim=2,
        )
