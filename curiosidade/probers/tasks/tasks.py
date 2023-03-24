"""Preconfigured probing tasks based on [1]_.

References
----------
.. [1] Alexis Conneau, German Kruszewski, Guillaume Lample, Loïc Barrault, and Marco Baroni.
   2018. What you can cram into a single $&!#* vector: Probing sentence embeddings for
   linguistic properties. In Proceedings of the 56th Annual Meeting of the Association for
   Computational Linguistics (Volume 1: Long Papers), pages 2126–2136, Melbourne, Australia.
   Association for Computational Linguistics.
"""
import typing as t
import inspect
import sys

import torch
import torch.nn

from . import base


__all__ = [
    "ProbingTaskSentenceLength",
    "ProbingTaskBigramShift",
    "ProbingTaskPastPresent",
    "ProbingTaskSOMO",
    "ProbingTaskSubjectNumber",
    "ProbingTaskObjectNumber",
    "ProbingTaskTreeDepth",
    "ProbingTaskCoordinationInversion",
    "ProbingTaskWordContent",
    "ProbingTaskTopConstituent",
    "ProbingTaskCustom",
    "get_available_preconfigured_tasks",
    "get_available_data_domains",
]


def get_available_preconfigured_tasks() -> tuple[tuple[str, base.BaseProbingTask], ...]:
    """Return all available preconfigured probing task classes."""
    tasks = tuple(
        inspect.getmembers(
            sys.modules[__name__],
            predicate=lambda x: inspect.isclass(x)
            and issubclass(x, base.BaseProbingTask)
            and x.__name__ != "ProbingTaskCustom",
        )
    )
    return tasks


def get_available_data_domains() -> tuple[str, ...]:
    """Return all available probing data domains."""
    return tuple(base.BaseProbingTask.VALID_DATA_DOMAINS)


class ProbingTaskSentenceLength(base.BaseProbingTask):
    """Preconfigured Sentence length probing task.

    Parameters
    ----------
    fn_raw_data_to_tensor : t.Callable[[list[str], list[int]], t.Any]
        Function used to transform raw data into PyTorch tensors. The output of this function
        will be feed directly into a `torch.utils.data.DataLoader`.

    batch_size_train : int, default=128
        Batch size for train dataloader.

    batch_size_eval : int, default=256
        Batch size for train validation and test dataloaders.

    data_domain : str, default='wikipedia-ptbr'
        Set the data domain for this probing task.

        - `wikipedia-ptbr`: General PT-br data domain from PT-br Wikipedia;
        - `sp-court-cases`: São Paulo (Brazil) Court cases;
        - `leg-docs-ptbr`: Brazilian legislative proposals;
        - `leg-pop-comments-ptbr`: Brazilian population comments regarding legislative proposals;
        - `political-speeches-ptbr`: Brazilian political speeches.

    output_dir : str, default='probing_datasets'
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

    timeout_limit_seconds : int, default=60
        Maximum time limit, in seconds, to try to download the probing dataset.
    """

    def __init__(
        self,
        fn_raw_data_to_tensor: t.Callable[[list[str], list[int]], t.Any],
        batch_size_train: int = 128,
        batch_size_eval: int = 256,
        data_domain: str = "wikipedia-ptbr",
        output_dir: str = "probing_datasets",
        metrics_fn: t.Optional[base.ValidationFunctionType] = None,
        show_progress_bar: bool = True,
        check_cached: bool = True,
        clean_compressed_files: bool = True,
        check_resource_hash: bool = True,
        timeout_limit_seconds: int = 60,
    ):
        self.check_if_domain_is_valid(data_domain)

        if data_domain == "wikipedia-ptbr":
            resource_name = "dataset_wikipedia_ptbr_sentence_length_v1"

        elif data_domain == "sp-court-cases":
            resource_name = "dataset_sp_court_cases_sentence_length_v1"

        elif data_domain == "leg-docs-ptbr":
            resource_name = "dataset_leg_docs_ptbr_sentence_length_v1"

        elif data_domain == "leg-pop-comments-ptbr":
            resource_name = "dataset_leg_pop_comments_ptbr_sentence_length_v1"

        elif data_domain == "political-speeches-ptbr":
            resource_name = "dataset_political_speeches_ptbr_sentence_length_v1"

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
            data_domain=data_domain,
            batch_size_train=batch_size_train,
            batch_size_eval=batch_size_eval,
            **resource_uris,
        )


class ProbingTaskWordContent(base.BaseProbingTask):
    """Preconfigured Word Content probing task.

    Parameters
    ----------
    fn_raw_data_to_tensor : t.Callable[[list[str], list[int]], t.Any]
        Function used to transform raw data into PyTorch tensors. The output of this function
        will be feed directly into a `torch.utils.data.DataLoader`.

    batch_size_train : int, default=128
        Batch size for train dataloader.

    batch_size_eval : int, default=256
        Batch size for train validation and test dataloaders.

    data_domain : str, default='wikipedia-ptbr'
        Set the data domain for this probing task.

        - `wikipedia-ptbr`: General PT-br data domain from PT-br Wikipedia;
        - `sp-court-cases`: São Paulo (Brazil) Court cases;
        - `leg-docs-ptbr`: Brazilian legislative proposals;
        - `leg-pop-comments-ptbr`: Brazilian population comments regarding legislative proposals;
        - `political-speeches-ptbr`: Brazilian political speeches.

    output_dir : str, default='probing_datasets'
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

    timeout_limit_seconds : int, default=60
        Maximum time limit, in seconds, to try to download the probing dataset.
    """

    def __init__(
        self,
        fn_raw_data_to_tensor: t.Callable[[list[str], list[int]], t.Any],
        batch_size_train: int = 128,
        batch_size_eval: int = 256,
        data_domain: str = "wikipedia-ptbr",
        output_dir: str = "probing_datasets",
        metrics_fn: t.Optional[base.ValidationFunctionType] = None,
        show_progress_bar: bool = True,
        check_cached: bool = True,
        clean_compressed_files: bool = True,
        check_resource_hash: bool = True,
        timeout_limit_seconds: int = 60,
    ):
        self.check_if_domain_is_valid(data_domain)

        if data_domain == "wikipedia-ptbr":
            resource_name = "dataset_wikipedia_ptbr_word_content_v1"

        elif data_domain == "sp-court-cases":
            resource_name = "dataset_sp_court_cases_word_content_v1"

        elif data_domain == "leg-docs-ptbr":
            resource_name = "dataset_leg_docs_ptbr_word_content_v1"

        elif data_domain == "leg-pop-comments-ptbr":
            resource_name = "dataset_leg_pop_comments_ptbr_word_content_v1"

        elif data_domain == "political-speeches-ptbr":
            resource_name = "dataset_political_speeches_ptbr_word_content_v1"

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
            data_domain=data_domain,
            batch_size_train=batch_size_train,
            batch_size_eval=batch_size_eval,
            **resource_uris,
        )


class ProbingTaskBigramShift(base.BaseProbingTask):
    """Preconfigured Bigram Shift probing task.

    Parameters
    ----------
    fn_raw_data_to_tensor : t.Callable[[list[str], list[int]], t.Any]
        Function used to transform raw data into PyTorch tensors. The output of this function
        will be feed directly into a `torch.utils.data.DataLoader`.

    batch_size_train : int, default=128
        Batch size for train dataloader.

    batch_size_eval : int, default=256
        Batch size for train validation and test dataloaders.

    data_domain : str, default='wikipedia-ptbr'
        Set the data domain for this probing task.

        - `wikipedia-ptbr`: General PT-br data domain from PT-br Wikipedia;
        - `sp-court-cases`: São Paulo (Brazil) Court cases;
        - `leg-docs-ptbr`: Brazilian legislative proposals;
        - `leg-pop-comments-ptbr`: Brazilian population comments regarding legislative proposals;
        - `political-speeches-ptbr`: Brazilian political speeches.

    output_dir : str, default='probing_datasets'
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

    timeout_limit_seconds : int, default=60
        Maximum time limit, in seconds, to try to download the probing dataset.
    """

    def __init__(
        self,
        fn_raw_data_to_tensor: t.Callable[[list[str], list[int]], t.Any],
        batch_size_train: int = 128,
        batch_size_eval: int = 256,
        data_domain: str = "wikipedia-ptbr",
        output_dir: str = "probing_datasets",
        metrics_fn: t.Optional[base.ValidationFunctionType] = None,
        show_progress_bar: bool = True,
        check_cached: bool = True,
        clean_compressed_files: bool = True,
        check_resource_hash: bool = True,
        timeout_limit_seconds: int = 60,
    ):
        self.check_if_domain_is_valid(data_domain)

        if data_domain == "wikipedia-ptbr":
            resource_name = "dataset_wikipedia_ptbr_bigram_shift_v1"

        elif data_domain == "sp-court-cases":
            resource_name = "dataset_sp_court_cases_bigram_shift_v1"

        elif data_domain == "leg-docs-ptbr":
            resource_name = "dataset_leg_docs_ptbr_bigram_shift_v1"

        elif data_domain == "leg-pop-comments-ptbr":
            resource_name = "dataset_leg_pop_comments_ptbr_bigram_shift_v1"

        elif data_domain == "political-speeches-ptbr":
            resource_name = "dataset_political_speeches_ptbr_bigram_shift_v1"

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
            data_domain=data_domain,
            batch_size_train=batch_size_train,
            batch_size_eval=batch_size_eval,
            **resource_uris,
        )


class ProbingTaskTreeDepth(base.BaseProbingTask):
    """Preconfigured Tree Depth probing task.

    Parameters
    ----------
    fn_raw_data_to_tensor : t.Callable[[list[str], list[int]], t.Any]
        Function used to transform raw data into PyTorch tensors. The output of this function
        will be feed directly into a `torch.utils.data.DataLoader`.

    batch_size_train : int, default=128
        Batch size for train dataloader.

    batch_size_eval : int, default=256
        Batch size for train validation and test dataloaders.

    data_domain : str, default='wikipedia-ptbr'
        Set the data domain for this probing task.

        - `wikipedia-ptbr`: General PT-br data domain from PT-br Wikipedia;
        - `sp-court-cases`: São Paulo (Brazil) Court cases;
        - `leg-docs-ptbr`: Brazilian legislative proposals;
        - `leg-pop-comments-ptbr`: Brazilian population comments regarding legislative proposals;
        - `political-speeches-ptbr`: Brazilian political speeches.

    output_dir : str, default='probing_datasets'
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

    timeout_limit_seconds : int, default=60
        Maximum time limit, in seconds, to try to download the probing dataset.
    """

    def __init__(
        self,
        fn_raw_data_to_tensor: t.Callable[[list[str], list[int]], t.Any],
        batch_size_train: int = 128,
        batch_size_eval: int = 256,
        data_domain: str = "wikipedia-ptbr",
        output_dir: str = "probing_datasets",
        metrics_fn: t.Optional[base.ValidationFunctionType] = None,
        show_progress_bar: bool = True,
        check_cached: bool = True,
        clean_compressed_files: bool = True,
        check_resource_hash: bool = True,
        timeout_limit_seconds: int = 60,
    ):
        self.check_if_domain_is_valid(data_domain)

        if data_domain == "wikipedia-ptbr":
            resource_name = "dataset_wikipedia_ptbr_tree_depth_v1"

        elif data_domain == "sp-court-cases":
            resource_name = "dataset_sp_court_cases_tree_depth_v1"

        elif data_domain == "leg-docs-ptbr":
            resource_name = "dataset_leg_docs_ptbr_tree_depth_v1"

        elif data_domain == "leg-pop-comments-ptbr":
            resource_name = "dataset_leg_pop_comments_ptbr_tree_depth_v1"

        elif data_domain == "political-speeches-ptbr":
            resource_name = "dataset_political_speeches_ptbr_tree_depth_v1"

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
            data_domain=data_domain,
            batch_size_train=batch_size_train,
            batch_size_eval=batch_size_eval,
            **resource_uris,
        )


class ProbingTaskTopConstituent(base.BaseProbingTask):
    """Preconfigured Top Constituent probing task.

    Parameters
    ----------
    fn_raw_data_to_tensor : t.Callable[[list[str], list[int]], t.Any]
        Function used to transform raw data into PyTorch tensors. The output of this function
        will be feed directly into a `torch.utils.data.DataLoader`.

    batch_size_train : int, default=128
        Batch size for train dataloader.

    batch_size_eval : int, default=256
        Batch size for train validation and test dataloaders.

    data_domain : str, default='wikipedia-ptbr'
        Set the data domain for this probing task.

        - `wikipedia-ptbr`: General PT-br data domain from PT-br Wikipedia;
        - `sp-court-cases`: São Paulo (Brazil) Court cases;
        - `leg-docs-ptbr`: Brazilian legislative proposals;
        - `leg-pop-comments-ptbr`: Brazilian population comments regarding legislative proposals;
        - `political-speeches-ptbr`: Brazilian political speeches.

    output_dir : str, default='probing_datasets'
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

    timeout_limit_seconds : int, default=60
        Maximum time limit, in seconds, to try to download the probing dataset.
    """

    def __init__(
        self,
        fn_raw_data_to_tensor: t.Callable[[list[str], list[int]], t.Any],
        batch_size_train: int = 128,
        batch_size_eval: int = 256,
        data_domain: str = "wikipedia-ptbr",
        output_dir: str = "probing_datasets",
        metrics_fn: t.Optional[base.ValidationFunctionType] = None,
        show_progress_bar: bool = True,
        check_cached: bool = True,
        clean_compressed_files: bool = True,
        check_resource_hash: bool = True,
        timeout_limit_seconds: int = 60,
    ):
        self.check_if_domain_is_valid(data_domain)

        if data_domain == "wikipedia-ptbr":
            resource_name = "dataset_wikipedia_ptbr_top_constituents_v1"

        elif data_domain == "sp-court-cases":
            resource_name = "dataset_sp_court_cases_top_constituents_v1"

        elif data_domain == "leg-docs-ptbr":
            resource_name = "dataset_leg_docs_ptbr_top_constituents_v1"

        elif data_domain == "leg-pop-comments-ptbr":
            resource_name = "dataset_leg_pop_comments_ptbr_top_constituents_v1"

        elif data_domain == "political-speeches-ptbr":
            resource_name = "dataset_political_speeches_ptbr_top_constituents_v1"

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
            data_domain=data_domain,
            batch_size_train=batch_size_train,
            batch_size_eval=batch_size_eval,
            **resource_uris,
        )


class ProbingTaskPastPresent(base.BaseProbingTask):
    """Preconfigured Past Present (or Sentence Tense) probing task.

    Parameters
    ----------
    fn_raw_data_to_tensor : t.Callable[[list[str], list[int]], t.Any]
        Function used to transform raw data into PyTorch tensors. The output of this function
        will be feed directly into a `torch.utils.data.DataLoader`.

    batch_size_train : int, default=128
        Batch size for train dataloader.

    batch_size_eval : int, default=256
        Batch size for train validation and test dataloaders.

    data_domain : str, default='wikipedia-ptbr'
        Set the data domain for this probing task.

        - `wikipedia-ptbr`: General PT-br data domain from PT-br Wikipedia;
        - `sp-court-cases`: São Paulo (Brazil) Court cases;
        - `leg-docs-ptbr`: Brazilian legislative proposals;
        - `leg-pop-comments-ptbr`: Brazilian population comments regarding legislative proposals;
        - `political-speeches-ptbr`: Brazilian political speeches.

    output_dir : str, default='probing_datasets'
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

    timeout_limit_seconds : int, default=60
        Maximum time limit, in seconds, to try to download the probing dataset.
    """

    def __init__(
        self,
        fn_raw_data_to_tensor: t.Callable[[list[str], list[int]], t.Any],
        batch_size_train: int = 128,
        batch_size_eval: int = 256,
        data_domain: str = "wikipedia-ptbr",
        output_dir: str = "probing_datasets",
        metrics_fn: t.Optional[base.ValidationFunctionType] = None,
        show_progress_bar: bool = True,
        check_cached: bool = True,
        clean_compressed_files: bool = True,
        check_resource_hash: bool = True,
        timeout_limit_seconds: int = 60,
    ):
        self.check_if_domain_is_valid(data_domain)

        if data_domain == "wikipedia-ptbr":
            resource_name = "dataset_wikipedia_ptbr_past_present_v1"

        elif data_domain == "sp-court-cases":
            resource_name = "dataset_sp_court_cases_past_present_v1"

        elif data_domain == "leg-docs-ptbr":
            resource_name = "dataset_leg_docs_ptbr_past_present_v1"

        elif data_domain == "leg-pop-comments-ptbr":
            resource_name = "dataset_leg_pop_comments_ptbr_past_present_v1"

        elif data_domain == "political-speeches-ptbr":
            resource_name = "dataset_political_speeches_ptbr_past_present_v1"

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
            data_domain=data_domain,
            batch_size_train=batch_size_train,
            batch_size_eval=batch_size_eval,
            **resource_uris,
        )


class ProbingTaskSubjectNumber(base.BaseProbingTask):
    """Preconfigured Subject Number probing task.

    Parameters
    ----------
    fn_raw_data_to_tensor : t.Callable[[list[str], list[int]], t.Any]
        Function used to transform raw data into PyTorch tensors. The output of this function
        will be feed directly into a `torch.utils.data.DataLoader`.

    batch_size_train : int, default=128
        Batch size for train dataloader.

    batch_size_eval : int, default=256
        Batch size for train validation and test dataloaders.

    data_domain : str, default='wikipedia-ptbr'
        Set the data domain for this probing task.

        - `wikipedia-ptbr`: General PT-br data domain from PT-br Wikipedia;
        - `sp-court-cases`: São Paulo (Brazil) Court cases;
        - `leg-docs-ptbr`: Brazilian legislative proposals;
        - `leg-pop-comments-ptbr`: Brazilian population comments regarding legislative proposals;
        - `political-speeches-ptbr`: Brazilian political speeches.

    output_dir : str, default='probing_datasets'
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

    timeout_limit_seconds : int, default=60
        Maximum time limit, in seconds, to try to download the probing dataset.
    """

    def __init__(
        self,
        fn_raw_data_to_tensor: t.Callable[[list[str], list[int]], t.Any],
        batch_size_train: int = 128,
        batch_size_eval: int = 256,
        data_domain: str = "wikipedia-ptbr",
        output_dir: str = "probing_datasets",
        metrics_fn: t.Optional[base.ValidationFunctionType] = None,
        show_progress_bar: bool = True,
        check_cached: bool = True,
        clean_compressed_files: bool = True,
        check_resource_hash: bool = True,
        timeout_limit_seconds: int = 60,
    ):
        self.check_if_domain_is_valid(data_domain)

        if data_domain == "wikipedia-ptbr":
            resource_name = "dataset_wikipedia_ptbr_subj_number_v1"

        elif data_domain == "sp-court-cases":
            resource_name = "dataset_sp_court_cases_subj_number_v1"

        elif data_domain == "leg-docs-ptbr":
            resource_name = "dataset_leg_docs_ptbr_subj_number_v1"

        elif data_domain == "leg-pop-comments-ptbr":
            resource_name = "dataset_leg_pop_comments_ptbr_subj_number_v1"

        elif data_domain == "political-speeches-ptbr":
            resource_name = "dataset_political_speeches_ptbr_subj_number_v1"

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
            data_domain=data_domain,
            batch_size_train=batch_size_train,
            batch_size_eval=batch_size_eval,
            **resource_uris,
        )


class ProbingTaskObjectNumber(base.BaseProbingTask):
    """Preconfigured Object Number probing task.

    Parameters
    ----------
    fn_raw_data_to_tensor : t.Callable[[list[str], list[int]], t.Any]
        Function used to transform raw data into PyTorch tensors. The output of this function
        will be feed directly into a `torch.utils.data.DataLoader`.

    batch_size_train : int, default=128
        Batch size for train dataloader.

    batch_size_eval : int, default=256
        Batch size for train validation and test dataloaders.

    data_domain : str, default='wikipedia-ptbr'
        Set the data domain for this probing task.

        - `wikipedia-ptbr`: General PT-br data domain from PT-br Wikipedia;
        - `sp-court-cases`: São Paulo (Brazil) Court cases;
        - `leg-docs-ptbr`: Brazilian legislative proposals;
        - `leg-pop-comments-ptbr`: Brazilian population comments regarding legislative proposals;
        - `political-speeches-ptbr`: Brazilian political speeches.

    output_dir : str, default='probing_datasets'
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

    timeout_limit_seconds : int, default=60
        Maximum time limit, in seconds, to try to download the probing dataset.
    """

    def __init__(
        self,
        fn_raw_data_to_tensor: t.Callable[[list[str], list[int]], t.Any],
        batch_size_train: int = 128,
        batch_size_eval: int = 256,
        data_domain: str = "wikipedia-ptbr",
        output_dir: str = "probing_datasets",
        metrics_fn: t.Optional[base.ValidationFunctionType] = None,
        show_progress_bar: bool = True,
        check_cached: bool = True,
        clean_compressed_files: bool = True,
        check_resource_hash: bool = True,
        timeout_limit_seconds: int = 60,
    ):
        self.check_if_domain_is_valid(data_domain)

        if data_domain == "wikipedia-ptbr":
            resource_name = "dataset_wikipedia_ptbr_obj_number_v1"

        elif data_domain == "sp-court-cases":
            resource_name = "dataset_sp_court_cases_obj_number_v1"

        elif data_domain == "leg-docs-ptbr":
            resource_name = "dataset_leg_docs_ptbr_obj_number_v1"

        elif data_domain == "leg-pop-comments-ptbr":
            resource_name = "dataset_leg_pop_comments_ptbr_obj_number_v1"

        elif data_domain == "political-speeches-ptbr":
            resource_name = "dataset_political_speeches_ptbr_obj_number_v1"

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
            data_domain=data_domain,
            batch_size_train=batch_size_train,
            batch_size_eval=batch_size_eval,
            **resource_uris,
        )


class ProbingTaskSOMO(base.BaseProbingTask):
    """Preconfigured Sentence Odd Man Out (SOMO) probing task.

    Parameters
    ----------
    fn_raw_data_to_tensor : t.Callable[[list[str], list[int]], t.Any]
        Function used to transform raw data into PyTorch tensors. The output of this function
        will be feed directly into a `torch.utils.data.DataLoader`.

    batch_size_train : int, default=128
        Batch size for train dataloader.

    batch_size_eval : int, default=256
        Batch size for train validation and test dataloaders.

    data_domain : str, default='wikipedia-ptbr'
        Set the data domain for this probing task.

        - `wikipedia-ptbr`: General PT-br data domain from PT-br Wikipedia;
        - `sp-court-cases`: São Paulo (Brazil) Court cases;
        - `leg-docs-ptbr`: Brazilian legislative proposals;
        - `leg-pop-comments-ptbr`: Brazilian population comments regarding legislative proposals;
        - `political-speeches-ptbr`: Brazilian political speeches.

    output_dir : str, default='probing_datasets'
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

    timeout_limit_seconds : int, default=60
        Maximum time limit, in seconds, to try to download the probing dataset.
    """

    def __init__(
        self,
        fn_raw_data_to_tensor: t.Callable[[list[str], list[int]], t.Any],
        batch_size_train: int = 128,
        batch_size_eval: int = 256,
        data_domain: str = "wikipedia-ptbr",
        output_dir: str = "probing_datasets",
        metrics_fn: t.Optional[base.ValidationFunctionType] = None,
        show_progress_bar: bool = True,
        check_cached: bool = True,
        clean_compressed_files: bool = True,
        check_resource_hash: bool = True,
        timeout_limit_seconds: int = 60,
    ):
        self.check_if_domain_is_valid(data_domain)

        if data_domain == "wikipedia-ptbr":
            resource_name = "dataset_wikipedia_ptbr_odd_man_out_v1"

        elif data_domain == "sp-court-cases":
            resource_name = "dataset_sp_court_cases_odd_man_out_v1"

        elif data_domain == "leg-docs-ptbr":
            resource_name = "dataset_leg_docs_ptbr_odd_man_out_v1"

        elif data_domain == "leg-pop-comments-ptbr":
            resource_name = "dataset_leg_pop_comments_ptbr_odd_man_out_v1"

        elif data_domain == "political-speeches-ptbr":
            resource_name = "dataset_political_speeches_ptbr_odd_man_out_v1"

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
            data_domain=data_domain,
            batch_size_train=batch_size_train,
            batch_size_eval=batch_size_eval,
            **resource_uris,
        )


class ProbingTaskCoordinationInversion(base.BaseProbingTask):
    """Preconfigured Coordination Inversion probing task.

    Parameters
    ----------
    fn_raw_data_to_tensor : t.Callable[[list[str], list[int]], t.Any]
        Function used to transform raw data into PyTorch tensors. The output of this function
        will be feed directly into a `torch.utils.data.DataLoader`.

    batch_size_train : int, default=128
        Batch size for train dataloader.

    batch_size_eval : int, default=256
        Batch size for train validation and test dataloaders.

    data_domain : str, default='wikipedia-ptbr'
        Set the data domain for this probing task.

        - `wikipedia-ptbr`: General PT-br data domain from PT-br Wikipedia;
        - `sp-court-cases`: São Paulo (Brazil) Court cases;
        - `leg-docs-ptbr`: Brazilian legislative proposals;
        - `leg-pop-comments-ptbr`: Brazilian population comments regarding legislative proposals;
        - `political-speeches-ptbr`: Brazilian political speeches.

    output_dir : str, default='probing_datasets'
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

    timeout_limit_seconds : int, default=60
        Maximum time limit, in seconds, to try to download the probing dataset.
    """

    def __init__(
        self,
        fn_raw_data_to_tensor: t.Callable[[list[str], list[int]], t.Any],
        batch_size_train: int = 128,
        batch_size_eval: int = 256,
        data_domain: str = "wikipedia-ptbr",
        output_dir: str = "probing_datasets",
        metrics_fn: t.Optional[base.ValidationFunctionType] = None,
        show_progress_bar: bool = True,
        check_cached: bool = True,
        clean_compressed_files: bool = True,
        check_resource_hash: bool = True,
        timeout_limit_seconds: int = 60,
    ):
        self.check_if_domain_is_valid(data_domain)

        if data_domain == "wikipedia-ptbr":
            resource_name = "dataset_wikipedia_ptbr_coordination_inversion_v1"

        elif data_domain == "sp-court-cases":
            resource_name = "dataset_sp_court_cases_coordination_inversion_v1"

        elif data_domain == "leg-docs-ptbr":
            resource_name = "dataset_leg_docs_ptbr_coordination_inversion_v1"

        elif data_domain == "leg-pop-comments-ptbr":
            resource_name = "dataset_leg_pop_comments_ptbr_coordination_inversion_v1"

        elif data_domain == "political-speeches-ptbr":
            resource_name = "dataset_political_speeches_ptbr_coordination_inversion_v1"

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
            data_domain=data_domain,
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

        - If str, assume it is an URI to a JSON file containing the mapping;
        - If sequence, assume that the sequence[i] corresponds to the `i`-th label;
        - If dict, assume that dict[label] = index;
        - If None, assume that labels are integers ranging from `0` to ``output_dim`` if \
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
