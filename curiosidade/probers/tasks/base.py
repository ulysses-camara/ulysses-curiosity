"""Base class for a probing task."""
import typing as t
import pathlib
import abc
import json
import os
import inspect

import torch
import torch.nn
import pandas as pd
import buscador

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
        and as the second argument a ground-truth label tensor of shape (batch_size,).
        The return value must always be a dictionary (or any other valid mapping) mapping the
        metric name and its computed value.
        If None, no extra validation metrics will be computed, and only the loss values will
        be returned as result.

    fn_raw_data_to_tensor : t.Callable[[list[str], list[int]], any] or None, default=None
        Function used to transform raw data into PyTorch tensors. The output of this function
        will be feed directly into a `torch.utils.data.DataLoader`. This argument is used
        only if any dataset `dataloader` is actually a file URI, as the data read from disk
        must be transformed into tensors.

    task_name : str, default="unnamed_task"
        Probing task name.

    task_type : {'classification', 'regression', 'mixed'}, default='classification'
        Type of task. Used only as reference, since it is the `loss_fn` that dictates
        how exactly the labels must be formatted.

    batch_size_train : int, default=64
        Batch size for train dataloader.

    batch_size_eval : int, default=128
        Batch size for eval and test dataloader (if any).
    """

    VALID_DATA_DOMAINS: t.Final[frozenset[str]] = frozenset(("wikipedia-ptbr", "sp-court-cases"))

    def __init__(
        self,
        loss_fn: LossFunctionType,
        labels_uri_or_map: t.Union[dict[str, int], t.Sequence[str], str],
        dataset_uri_or_dataloader_train: DataLoaderGenericType,
        dataset_uri_or_dataloader_eval: t.Optional[DataLoaderGenericType] = None,
        dataset_uri_or_dataloader_test: t.Optional[DataLoaderGenericType] = None,
        output_dim: t.Union[int, t.Literal["infer_from_labels"]] = "infer_from_labels",
        metrics_fn: t.Optional[ValidationFunctionType] = None,
        fn_raw_data_to_tensor: t.Optional[t.Callable[[list[str], list[int]], t.Any]] = None,
        task_name: str = "unnamed_task",
        task_type: t.Literal["classification", "regression", "mixed"] = "classification",
        batch_size_train: int = 64,
        batch_size_eval: int = 128,
    ):
        if task_type not in {"classification", "regression", "mixed"}:
            raise ValueError(
                f"Provided 'task_type' unsupported ('{task_type}'). Must be either "
                "'classification', 'regression' or 'mixed'."
            )

        self.task_name = task_name
        self.metrics_fn = metrics_fn
        self.fn_raw_data_to_tensor = fn_raw_data_to_tensor
        self.loss_fn = loss_fn
        self.task_type = task_type

        dl_train: DataLoaderType
        dl_eval: t.Optional[DataLoaderType]
        dl_test: t.Optional[DataLoaderType]
        self.labels: t.Dict[str, int]

        if isinstance(labels_uri_or_map, dict):
            self.labels = labels_uri_or_map.copy()

        elif isinstance(labels_uri_or_map, (str, pathlib.Path)):
            with open(labels_uri_or_map, "r", encoding="utf-8") as f_in:
                labels = json.load(f_in)[self.task_name]

            self.labels = {cls: ind for ind, cls in enumerate(labels)}

        else:
            self.labels = {cls: ind for ind, cls in enumerate(labels_uri_or_map)}

        if output_dim == "infer_from_labels":
            output_dim = len(self.labels) if len(self.labels) >= 3 else 1

        output_dim = int(output_dim)

        if output_dim <= 0:
            raise ValueError(f"Invalid 'output_dim' ({output_dim}), must be >= 1.")

        self.output_dim = output_dim

        if isinstance(dataset_uri_or_dataloader_train, (str, pathlib.Path)):
            dl_train = torch.utils.data.DataLoader(
                self._load_dataset(dataset_uri_or_dataloader_train, split="train"),
                batch_size=batch_size_train or 64,
                shuffle=True,
            )

        else:
            dl_train = dataset_uri_or_dataloader_train

        if isinstance(dataset_uri_or_dataloader_eval, (str, pathlib.Path)):
            dl_eval = torch.utils.data.DataLoader(
                self._load_dataset(dataset_uri_or_dataloader_eval, split="eval"),
                batch_size=batch_size_eval or 128,
                shuffle=False,
            )

        else:
            dl_eval = dataset_uri_or_dataloader_eval

        if isinstance(dataset_uri_or_dataloader_test, (str, pathlib.Path)):
            dl_test = torch.utils.data.DataLoader(
                self._load_dataset(dataset_uri_or_dataloader_test, split="test"),
                batch_size=batch_size_eval or 128,
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
    ) -> t.Any:
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

        if not self.fn_raw_data_to_tensor:
            raise ValueError(
                "The function 'fn_raw_data_to_tensor' is not provided, but it is required to "
                "appropriately transform raw data loaded from disk to PyTorch tensors. "
                "Please either provide 'fn_raw_data_to_tensor' or a preloaded "
                "'torch.utils.data.DataLoader' in 'dataset_uri_or_dataloader_train' argument "
                "(and optionally 'dataset_uri_or_dataloader_eval' and "
                "'dataset_uri_or_dataloader_test')."
            )

        if "split" in inspect.signature(self.fn_raw_data_to_tensor).parameters:
            out = self.fn_raw_data_to_tensor(content, labels, split=split)  # type: ignore

        else:
            out = self.fn_raw_data_to_tensor(content, labels)

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
            fn_raw_data_to_tensor=lambda *args, **kwargs: None,
            labels_uri_or_map=["A", "B"],
            output_dim=2,
        )


def get_resource_from_ulysses_fetcher(
    resource_name: str, output_dir: str, **kwargs: t.Any
) -> dict[str, str]:
    """Download necessary resources using Ulysses Fetcher.

    Parameters
    ----------
    resource_name : str
        Requested resource name. See [1]_ for available resources for `probing_task` task.

    output_dir : str
        Output directory to store downloaded resources.

    **kwargs : any
        Additional named parameters for ``buscador.download_resource`` function.
        Check Ulysses Fetcher documentation for more information.

    References
    ----------
    .. [1] Ulysses Fetcher: https://github.com/ulysses-camara/ulysses-fetcher
    """
    has_succeed = buscador.download_resource(
        task_name="probing_task",
        resource_name=resource_name,
        output_dir=output_dir,
        **kwargs,
    )

    if not has_succeed:
        raise FileNotFoundError(
            "Could not find or download the necessary resource '{resource_name}' using the "
            "Ulysses Fetcher. Please check write permissions or connectivity issues."
        )

    input_dir = os.path.join(output_dir, resource_name)
    input_dir = os.path.abspath(input_dir)

    dataset_uri_train = os.path.join(input_dir, "train.tsv")
    dataset_uri_eval = os.path.join(input_dir, "eval.tsv")
    dataset_uri_test = os.path.join(input_dir, "test.tsv")
    labels_uri = os.path.join(input_dir, "labels.json")

    resources_uris = dict(
        dataset_uri_or_dataloader_train=dataset_uri_train,
        dataset_uri_or_dataloader_eval=dataset_uri_eval,
        dataset_uri_or_dataloader_test=dataset_uri_test,
        labels_uri_or_map=labels_uri,
    )

    return resources_uris
