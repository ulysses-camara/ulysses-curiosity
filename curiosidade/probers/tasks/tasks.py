"""Probing task classes."""
import typing as t

import torch
import torch.nn

from . import base


class ProbingTaskSentenceLength(base.BaseProbingTask):
    """Preconfigured Sentence length (SentLen) probing task.

    Based on [1]_.

    Parameters
    ----------
    fn_raw_data_to_tensor : t.Callable[[list[str], list[int]], any]
        Function used to transform raw data into PyTorch tensors. The output of this function
        will be feed directly into a `torch.utils.data.DataLoader`.

    batch_size_train : int, default=128
        Batch size for train dataloader.

    batch_size_eval : int, default=256
        Batch size for train validation and test dataloaders.

    data_domain : {"general-pt-br"}, default="general-pt-br"
        Set the data domain for this probing task.

        - `general-pt-br`: General PT-br data domain from Portuguese Wikipedia.

    output_dir : str, default="probing_datasets"
        Output directory for probing datasets.

    metrics_fn : base.ValidationFunctionType or None, default=None
        Validation function to compute extra scores from training, validation and test batches.
        As the first argument, it must receive a logit tensor of shape (batch_size, output_dim),
        and as the second argument a ground-truth label tensor of shape (batch_size,).
        The return value must always be a dictionary (or any other valid mapping) mapping the
        metric name and its computed value.
        If None, no extra validation metrics will be computed, and only the loss values will
        be returned as result.

    show_progress_bar : bool, default=True
        If True, show progress bar while downloading probing datasets.

    check_cached : bool, default=True
        If True, check if probing datasets are available locally before downloading.

    clean_compressed_files : bool, default=True
        If True, delete compressed probing datasets after decompression.

    check_resource_hash : bool, default=True
        If True, verify downloaded probing dataset hash.

    timeout_limit_seconds : int, default=10
        Maximum time limit, in seconds, to try to download the probing dataset.

    References
    ----------
    .. [1] Alexis Conneau, German Kruszewski, Guillaume Lample, Loïc Barrault, and Marco Baroni.
       2018. What you can cram into a single $&!#* vector: Probing sentence embeddings for
       linguistic properties. In Proceedings of the 56th Annual Meeting of the Association for
       Computational Linguistics (Volume 1: Long Papers), pages 2126–2136, Melbourne, Australia.
       Association for Computational Linguistics.
    .. [2] Ulysses Fetcher: https://github.com/ulysses-camara/ulysses-fetcher
    """

    def __init__(
        self,
        fn_raw_data_to_tensor: t.Callable[[list[str], list[int]], t.Any],
        batch_size_train: int = 128,
        batch_size_eval: int = 256,
        data_domain: str = "general-pt-br",
        output_dir: str = "probing_datasets",
        metrics_fn: t.Optional[base.ValidationFunctionType] = None,
        show_progress_bar: bool = True,
        check_cached: bool = True,
        clean_compressed_files: bool = True,
        check_resource_hash: bool = True,
        timeout_limit_seconds: int = 10,
    ):
        self.check_if_domain_is_valid(data_domain)

        if data_domain == "general-pt-br":
            resource_name = "dataset_wikipedia_ptbr_sentence_length_v1"

        resource_uris = base.get_resource_from_ulysses_fetcher(
            resource_name=resource_name,
            output_dir=output_dir,
            show_progress_bar=show_progress_bar,
            check_cached=check_cached,
            clean_compressed_files=clean_compressed_files,
            check_resource_hash=check_resource_hash,
            timeout_limit_seconds=timeout_limit_seconds,
        )

        super().__init__(
            loss_fn=torch.nn.CrossEntropyLoss(),
            metrics_fn=metrics_fn,
            fn_raw_data_to_tensor=fn_raw_data_to_tensor,
            output_dim="infer_from_labels",
            task_type="classification",
            task_name="sentence_length",
            batch_size_train=batch_size_train,
            batch_size_eval=batch_size_eval,
            **resource_uris,
        )


class ProbingTaskWordContent(base.BaseProbingTask):
    """TODO"""

    def __init__(
        self,
        fn_raw_data_to_tensor: t.Callable[[list[str], list[int]], t.Any],
        batch_size_train: int = 128,
        batch_size_eval: int = 256,
        data_domain: str = "general-pt-br",
        output_dir: str = "probing_datasets",
        metrics_fn: t.Optional[base.ValidationFunctionType] = None,
        show_progress_bar: bool = True,
        check_cached: bool = True,
        clean_compressed_files: bool = True,
        check_resource_hash: bool = True,
        timeout_limit_seconds: int = 10,
    ):
        self.check_if_domain_is_valid(data_domain)

        if data_domain == "general-pt-br":
            resource_name = "dataset_wikipedia_ptbr_word_content_v1"

        resource_uris = base.get_resource_from_ulysses_fetcher(
            resource_name=resource_name,
            output_dir=output_dir,
            show_progress_bar=show_progress_bar,
            check_cached=check_cached,
            clean_compressed_files=clean_compressed_files,
            check_resource_hash=check_resource_hash,
            timeout_limit_seconds=timeout_limit_seconds,
        )

        super().__init__(
            loss_fn=torch.nn.CrossEntropyLoss(),
            metrics_fn=metrics_fn,
            fn_raw_data_to_tensor=fn_raw_data_to_tensor,
            output_dim="infer_from_labels",
            task_type="classification",
            task_name="word_content",
            batch_size_train=batch_size_train,
            batch_size_eval=batch_size_eval,
            **resource_uris,
        )


class ProbingTaskBigramShift(base.BaseProbingTask):
    """TODO"""

    def __init__(
        self,
        fn_raw_data_to_tensor: t.Callable[[list[str], list[int]], t.Any],
        batch_size_train: int = 128,
        batch_size_eval: int = 256,
        data_domain: str = "general-pt-br",
        output_dir: str = "probing_datasets",
        metrics_fn: t.Optional[base.ValidationFunctionType] = None,
        show_progress_bar: bool = True,
        check_cached: bool = True,
        clean_compressed_files: bool = True,
        check_resource_hash: bool = True,
        timeout_limit_seconds: int = 10,
    ):
        self.check_if_domain_is_valid(data_domain)

        if data_domain == "general-pt-br":
            resource_name = "dataset_wikipedia_ptbr_bigram_shift_v1"

        resource_uris = base.get_resource_from_ulysses_fetcher(
            resource_name=resource_name,
            output_dir=output_dir,
            show_progress_bar=show_progress_bar,
            check_cached=check_cached,
            clean_compressed_files=clean_compressed_files,
            check_resource_hash=check_resource_hash,
            timeout_limit_seconds=timeout_limit_seconds,
        )

        super().__init__(
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            metrics_fn=metrics_fn,
            fn_raw_data_to_tensor=fn_raw_data_to_tensor,
            output_dim=1,
            task_type="classification",
            task_name="bigram_shift",
            batch_size_train=batch_size_train,
            batch_size_eval=batch_size_eval,
            **resource_uris,
        )


class ProbingTaskTreeDepth(base.BaseProbingTask):
    """TODO"""

    def __init__(
        self,
        fn_raw_data_to_tensor: t.Callable[[list[str], list[int]], t.Any],
        batch_size_train: int = 128,
        batch_size_eval: int = 256,
        data_domain: str = "general-pt-br",
        output_dir: str = "probing_datasets",
        metrics_fn: t.Optional[base.ValidationFunctionType] = None,
        show_progress_bar: bool = True,
        check_cached: bool = True,
        clean_compressed_files: bool = True,
        check_resource_hash: bool = True,
        timeout_limit_seconds: int = 10,
    ):
        self.check_if_domain_is_valid(data_domain)

        if data_domain == "general-pt-br":
            resource_name = "dataset_wikipedia_ptbr_tree_depth_v1"

        resource_uris = base.get_resource_from_ulysses_fetcher(
            resource_name=resource_name,
            output_dir=output_dir,
            show_progress_bar=show_progress_bar,
            check_cached=check_cached,
            clean_compressed_files=clean_compressed_files,
            check_resource_hash=check_resource_hash,
            timeout_limit_seconds=timeout_limit_seconds,
        )

        super().__init__(
            loss_fn=torch.nn.CrossEntropyLoss(),
            metrics_fn=metrics_fn,
            fn_raw_data_to_tensor=fn_raw_data_to_tensor,
            output_dim="infer_from_labels",
            task_type="classification",
            task_name="tree_depth",
            batch_size_train=batch_size_train,
            batch_size_eval=batch_size_eval,
            **resource_uris,
        )


class ProbingTaskTopConstituent(base.BaseProbingTask):
    """TODO"""

    def __init__(
        self,
        fn_raw_data_to_tensor: t.Callable[[list[str], list[int]], t.Any],
        batch_size_train: int = 128,
        batch_size_eval: int = 256,
        data_domain: str = "general-pt-br",
        output_dir: str = "probing_datasets",
        metrics_fn: t.Optional[base.ValidationFunctionType] = None,
        show_progress_bar: bool = True,
        check_cached: bool = True,
        clean_compressed_files: bool = True,
        check_resource_hash: bool = True,
        timeout_limit_seconds: int = 10,
    ):
        self.check_if_domain_is_valid(data_domain)

        if data_domain == "general-pt-br":
            resource_name = "dataset_wikipedia_ptbr_top_constituents_v1"

        resource_uris = base.get_resource_from_ulysses_fetcher(
            resource_name=resource_name,
            output_dir=output_dir,
            show_progress_bar=show_progress_bar,
            check_cached=check_cached,
            clean_compressed_files=clean_compressed_files,
            check_resource_hash=check_resource_hash,
            timeout_limit_seconds=timeout_limit_seconds,
        )

        super().__init__(
            loss_fn=torch.nn.CrossEntropyLoss(),
            metrics_fn=metrics_fn,
            fn_raw_data_to_tensor=fn_raw_data_to_tensor,
            output_dim="infer_from_labels",
            task_type="classification",
            task_name="top_constituents",
            batch_size_train=batch_size_train,
            batch_size_eval=batch_size_eval,
            **resource_uris,
        )


class ProbingTaskPastPresent(base.BaseProbingTask):
    """TODO"""

    def __init__(
        self,
        fn_raw_data_to_tensor: t.Callable[[list[str], list[int]], t.Any],
        batch_size_train: int = 128,
        batch_size_eval: int = 256,
        data_domain: str = "general-pt-br",
        output_dir: str = "probing_datasets",
        metrics_fn: t.Optional[base.ValidationFunctionType] = None,
        show_progress_bar: bool = True,
        check_cached: bool = True,
        clean_compressed_files: bool = True,
        check_resource_hash: bool = True,
        timeout_limit_seconds: int = 10,
    ):
        self.check_if_domain_is_valid(data_domain)

        if data_domain == "general-pt-br":
            resource_name = "dataset_wikipedia_ptbr_past_present_v1"

        resource_uris = base.get_resource_from_ulysses_fetcher(
            resource_name=resource_name,
            output_dir=output_dir,
            show_progress_bar=show_progress_bar,
            check_cached=check_cached,
            clean_compressed_files=clean_compressed_files,
            check_resource_hash=check_resource_hash,
            timeout_limit_seconds=timeout_limit_seconds,
        )

        super().__init__(
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            metrics_fn=metrics_fn,
            fn_raw_data_to_tensor=fn_raw_data_to_tensor,
            output_dim=1,
            task_type="classification",
            task_name="past_present",
            batch_size_train=batch_size_train,
            batch_size_eval=batch_size_eval,
            **resource_uris,
        )


class ProbingTaskSubjectNumber(base.BaseProbingTask):
    """TODO"""

    def __init__(
        self,
        fn_raw_data_to_tensor: t.Callable[[list[str], list[int]], t.Any],
        batch_size_train: int = 128,
        batch_size_eval: int = 256,
        data_domain: str = "general-pt-br",
        output_dir: str = "probing_datasets",
        metrics_fn: t.Optional[base.ValidationFunctionType] = None,
        show_progress_bar: bool = True,
        check_cached: bool = True,
        clean_compressed_files: bool = True,
        check_resource_hash: bool = True,
        timeout_limit_seconds: int = 10,
    ):
        self.check_if_domain_is_valid(data_domain)

        if data_domain == "general-pt-br":
            resource_name = "dataset_wikipedia_ptbr_subj_number_v1"

        resource_uris = base.get_resource_from_ulysses_fetcher(
            resource_name=resource_name,
            output_dir=output_dir,
            show_progress_bar=show_progress_bar,
            check_cached=check_cached,
            clean_compressed_files=clean_compressed_files,
            check_resource_hash=check_resource_hash,
            timeout_limit_seconds=timeout_limit_seconds,
        )

        super().__init__(
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            metrics_fn=metrics_fn,
            fn_raw_data_to_tensor=fn_raw_data_to_tensor,
            output_dim=1,
            task_type="classification",
            task_name="subj_number",
            batch_size_train=batch_size_train,
            batch_size_eval=batch_size_eval,
            **resource_uris,
        )


class ProbingTaskObjectNumber(base.BaseProbingTask):
    """TODO"""

    def __init__(
        self,
        fn_raw_data_to_tensor: t.Callable[[list[str], list[int]], t.Any],
        batch_size_train: int = 128,
        batch_size_eval: int = 256,
        data_domain: str = "general-pt-br",
        output_dir: str = "probing_datasets",
        metrics_fn: t.Optional[base.ValidationFunctionType] = None,
        show_progress_bar: bool = True,
        check_cached: bool = True,
        clean_compressed_files: bool = True,
        check_resource_hash: bool = True,
        timeout_limit_seconds: int = 10,
    ):
        self.check_if_domain_is_valid(data_domain)

        if data_domain == "general-pt-br":
            resource_name = "dataset_wikipedia_ptbr_obj_number_v1"

        resource_uris = base.get_resource_from_ulysses_fetcher(
            resource_name=resource_name,
            output_dir=output_dir,
            show_progress_bar=show_progress_bar,
            check_cached=check_cached,
            clean_compressed_files=clean_compressed_files,
            check_resource_hash=check_resource_hash,
            timeout_limit_seconds=timeout_limit_seconds,
        )

        super().__init__(
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            metrics_fn=metrics_fn,
            fn_raw_data_to_tensor=fn_raw_data_to_tensor,
            output_dim=1,
            task_type="classification",
            task_name="obj_number",
            batch_size_train=batch_size_train,
            batch_size_eval=batch_size_eval,
            **resource_uris,
        )


class ProbingTaskSOMO(base.BaseProbingTask):
    """TODO"""

    def __init__(
        self,
        fn_raw_data_to_tensor: t.Callable[[list[str], list[int]], t.Any],
        batch_size_train: int = 128,
        batch_size_eval: int = 256,
        data_domain: str = "general-pt-br",
        output_dir: str = "probing_datasets",
        metrics_fn: t.Optional[base.ValidationFunctionType] = None,
        show_progress_bar: bool = True,
        check_cached: bool = True,
        clean_compressed_files: bool = True,
        check_resource_hash: bool = True,
        timeout_limit_seconds: int = 10,
    ):
        self.check_if_domain_is_valid(data_domain)

        if data_domain == "general-pt-br":
            resource_name = "dataset_wikipedia_ptbr_odd_man_out_v1"

        resource_uris = base.get_resource_from_ulysses_fetcher(
            resource_name=resource_name,
            output_dir=output_dir,
            show_progress_bar=show_progress_bar,
            check_cached=check_cached,
            clean_compressed_files=clean_compressed_files,
            check_resource_hash=check_resource_hash,
            timeout_limit_seconds=timeout_limit_seconds,
        )

        super().__init__(
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            metrics_fn=metrics_fn,
            fn_raw_data_to_tensor=fn_raw_data_to_tensor,
            output_dim=1,
            task_type="classification",
            task_name="odd_man_out",
            batch_size_train=batch_size_train,
            batch_size_eval=batch_size_eval,
            **resource_uris,
        )


class ProbingTaskCoordinationInversion(base.BaseProbingTask):
    """TODO"""

    def __init__(
        self,
        fn_raw_data_to_tensor: t.Callable[[list[str], list[int]], t.Any],
        batch_size_train: int = 128,
        batch_size_eval: int = 256,
        data_domain: str = "general-pt-br",
        output_dir: str = "probing_datasets",
        metrics_fn: t.Optional[base.ValidationFunctionType] = None,
        show_progress_bar: bool = True,
        check_cached: bool = True,
        clean_compressed_files: bool = True,
        check_resource_hash: bool = True,
        timeout_limit_seconds: int = 10,
    ):
        self.check_if_domain_is_valid(data_domain)

        if data_domain == "general-pt-br":
            resource_name = "dataset_wikipedia_ptbr_coordination_inversion_v1"

        resource_uris = base.get_resource_from_ulysses_fetcher(
            resource_name=resource_name,
            output_dir=output_dir,
            show_progress_bar=show_progress_bar,
            check_cached=check_cached,
            clean_compressed_files=clean_compressed_files,
            check_resource_hash=check_resource_hash,
            timeout_limit_seconds=timeout_limit_seconds,
        )

        super().__init__(
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            metrics_fn=metrics_fn,
            fn_raw_data_to_tensor=fn_raw_data_to_tensor,
            output_dim=1,
            task_type="classification",
            task_name="coordination_inversion",
            batch_size_train=batch_size_train,
            batch_size_eval=batch_size_eval,
            **resource_uris,
        )


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

    metrics_fn : t.Callable[[torch.Tensor, torch.Tensor], dict[str, float]] or None,\
            default=None
        Validation function to compute extra scores from training, validation and test batches.
        As the first argument, it must receive a logit tensor of shape (batch_size, output_dim),
        and  a ground-truth label tensor os shape (batch_size,) as the second argument.
        The return value must always be a dictionary (or any other valid mapping) mapping the
        metric name and its computed value.
        If None, no extra validation metrics will be computed, and only the loss values will
        be returned as result.

    labels_uri_or_map : str or t.Sequence[str] or dict[str, int] or None, default=None
        Map labels to indices.

        If str, assume it is an URI to a JSON file containing the mapping;
        If sequence, assume that the sequence[i] corresponds to the `i`-th label;
        If dict, assume that dict[label] = index;
        If None, assume that labels are integers ranging from `0` to ``output_dim`` if \
                ``output_dim >= 2``, else [0, 1].

    task_name : str, default='unnamed_task'
        Probing task name.

    task_type : {'classification', 'regression', 'mixed'}, default='classification'
        Type of task. Used only as reference, since it is the `loss_fn` that dictates
        how exactly the labels must be formatted.
    """

    def __init__(
        self,
        output_dim: int,
        loss_fn: base.LossFunctionType,
        probing_dataloader_train: base.DataLoaderType,
        probing_dataloader_eval: t.Optional[base.DataLoaderType] = None,
        probing_dataloader_test: t.Optional[base.DataLoaderType] = None,
        metrics_fn: t.Optional[base.ValidationFunctionType] = None,
        labels_uri_or_map: t.Optional[t.Union[str, t.Sequence[str], dict[str, int]]] = None,
        task_name: str = "unnamed_task",
        task_type: t.Literal["classification", "regression", "mixed"] = "classification",
    ):
        if labels_uri_or_map is None:
            labels_uri_or_map = list(map(str, range(max(2, output_dim))))

        if not isinstance(probing_dataloader_train, torch.utils.data.DataLoader):
            raise TypeError(
                "You must provide a PyTorch dataloader in 'probing_dataloader_train' argument "
                f"(got '{type(probing_dataloader_train) = }')."
            )

        for dloader in (probing_dataloader_eval, probing_dataloader_test):
            if dloader is not None and not isinstance(dloader, torch.utils.data.DataLoader):
                raise TypeError(
                    "Custom probing tasks only accept dataloaders as train, eval and test data "
                    f"(got '{type(dloader) = }')."
                )

        super().__init__(
            dataset_uri_or_dataloader_train=probing_dataloader_train,
            dataset_uri_or_dataloader_eval=probing_dataloader_eval,
            dataset_uri_or_dataloader_test=probing_dataloader_test,
            labels_uri_or_map=labels_uri_or_map,
            loss_fn=loss_fn,
            metrics_fn=metrics_fn,
            output_dim=output_dim,
            task_name=task_name,
            task_type=task_type,
        )
