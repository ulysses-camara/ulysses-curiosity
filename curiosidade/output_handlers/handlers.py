"""Handle probing results appropriately."""
import typing as t

import pandas as pd
import numpy as np


class MetricPack:
    """Store and aggregate metrics from probe model training."""

    def __init__(self) -> None:
        self.stored_values: dict[tuple[t.Any, ...], float] = {}

    def __getitem__(self, key: tuple[t.Any, ...]) -> float:
        return self.stored_values[key]

    def __iter__(self) -> t.Iterator[tuple[tuple[t.Any, ...], float]]:
        return iter(self.stored_values.items())

    def __repr__(self) -> str:
        stored_val_count = len(self.stored_values)
        num_print_each_half = 3

        pieces: list[str] = [f"MetricPack with {stored_val_count} values stored in:"]

        for i, (key, val) in enumerate(self.stored_values.items()):
            if num_print_each_half <= i < stored_val_count - num_print_each_half:
                continue

            pieces.append(f"  {key}: {val}")

            if i == num_print_each_half - 1 and len(self.stored_values) > 2 * num_print_each_half:
                pieces.append("  ...")

        return "\n".join(pieces)

    def append(
        self, metrics: dict[t.Any, float], *args: t.Any, merge_keys: bool = True
    ) -> "MetricPack":
        """Append extra metrics in stored items.

        Parameters
        ----------
        metrics : dict[t.Any, float]
            Metrics to store.

        *args : tuple[t.Any]
            Extra arguments to build the keys.

        merge_keys : bool, default=True
            If True, colapse containers within existing keys in `metrics` to merge with
            provided *args (if any). If False, the new keys will be a 2-tuple in the
            format (previous_key, *args).

        Returns
        -------
        self
        """
        for key, val in metrics.items():
            do_merge = merge_keys and hasattr(key, "__len__") and not isinstance(key, str)
            new_key = (*key, *args) if do_merge else (key, *args)
            self.stored_values[new_key] = val

        return self

    def expand_key_dim(self, *args: t.Any) -> "MetricPack":
        """Add a new value at the start of every stored key."""
        new_stored_values: dict[tuple[t.Any, ...], float] = {}

        for key, val in self.stored_values.items():
            new_key = (*args, *key)
            new_stored_values[new_key] = val

        self.stored_values = new_stored_values

        return self

    def combine(self, metrics: "MetricPack") -> "MetricPack":
        """Combine stored items with items in `metrics`."""
        self.stored_values.update(metrics.stored_values)
        return self

    def to_pandas(
        self,
        aggregate_by: t.Optional[t.Sequence[str]] = None,
        aggregate_fn: t.Callable[[t.Sequence[float]], float] = np.mean,
    ) -> pd.DataFrame:
        """Build a pandas DataFrame with stored values.

        Parameters
        ----------
        aggregate_by : t.Sequence[str] or None, default=None
            If given, aggregate results by the provided keys. It must be a sequence of strings
            contained a subset of {'epoch', 'metric_name', 'module', 'batch_index'}. The function
            used to aggregate values that fall in the same bucket is `aggregate_fn`.

        aggregate_fn : callable or sequence of callables, default=numpy.mean
            Function, or sequence of callables used to aggregate values in a same bucket. Also
            accept values supported by pandas Aggregate function. Must receive a sequence of
            values, and return a single number. Ignored if `aggregate_by=None`.

        Returns
        -------
        dataframe : pandas.DataFrame
            Stored values in the form of a pandas dataframe.

        Examples
        --------
        >>> import curiosidade
        ...
        >>> metrics = curiosidade.output_handlers.MetricPack()
        >>> metrics.append({
        ...     (0, 'loss', 'relu1', 0): 1.50,
        ...     (0,  'acc', 'relu1', 0): 0.10,
        ...     (0, 'loss', 'relu1', 1): 1.40,
        ...     (0,  'acc', 'relu1', 1): 0.20,
        ...     (1, 'loss', 'relu1', 0): 1.15,
        ...     (1,  'acc', 'relu1', 0): 0.33,
        ...     (1, 'loss', 'relu1', 1): 0.85,
        ...     (1,  'acc', 'relu1', 1): 0.43,
        ... })
        MetricPack with 8 values stored in:
          (0, 'loss', 'relu1', 0): 1.5
          (0, 'acc', 'relu1', 0): 0.1
          (0, 'loss', 'relu1', 1): 1.4
          ...
          (1, 'acc', 'relu1', 0): 0.33
          (1, 'loss', 'relu1', 1): 0.85
          (1, 'acc', 'relu1', 1): 0.43
        >>> metrics.to_pandas(
        ...     aggregate_by=['batch_index'],
        ...     aggregate_fn=[np.max, np.min],
        ... ).values
        array([[0, 'acc', 'relu1', 0.2, 0.1],
               [0, 'loss', 'relu1', 1.5, 1.4],
               [1, 'acc', 'relu1', 0.43, 0.33],
               [1, 'loss', 'relu1', 1.15, 0.85]], dtype=object)
        """
        dataframe = pd.DataFrame(
            [[*keys, val] for keys, val in self.stored_values.items()],
            columns=["epoch", "metric_name", "module", "batch_index", "metric"],
        )

        if aggregate_by is not None:
            aggregate_by_set = set(aggregate_by)

            if "metric" in aggregate_by_set:
                raise ValueError("Can not aggregate by 'metric'.")

            if not aggregate_by_set.issubset(dataframe.columns):
                df_columns_set = set(dataframe.columns)
                df_columns_set.remove("metric")

                unknown_cols = tuple(sorted(aggregate_by_set - df_columns_set))

                raise ValueError(
                    f"Unknown aggregation columns: {unknown_cols}. Must be a combination of the "
                    f"following: {sorted(tuple(df_columns_set))}."
                )

            group_by_set = sorted(set(dataframe.columns) - aggregate_by_set - {"metric"})

            dataframe = dataframe.groupby(group_by_set).agg(dict(metric=aggregate_fn))
            dataframe.reset_index(inplace=True)

        return dataframe


class ProbingResults(t.NamedTuple):
    """Class to store probing results per split."""

    train: t.Union[pd.DataFrame, MetricPack]
    eval: t.Optional[t.Union[pd.DataFrame, MetricPack]] = None
    test: t.Optional[t.Union[pd.DataFrame, MetricPack]] = None

    def _format_split(self, split_name: str) -> str:
        split = getattr(self, split_name)
        prefix = f"{split_name:}"
        prefix = f"    {prefix:<5} = "
        split_str = str(split).replace("\n", "\n  ")
        return f"{prefix}{split_str},"

    def __repr__(self) -> str:
        pieces: list[str] = []

        pieces.append(f"{self.__class__.__name__}(")
        pieces.append(self._format_split("train"))
        pieces.append(self._format_split("eval"))
        pieces.append(self._format_split("test"))
        pieces.append(")")

        return "\n".join(pieces)

    def to_pandas(
        self,
        aggregate_by: t.Optional[t.Sequence[str]] = None,
        aggregate_fn: t.Callable[[t.Sequence[float]], float] = np.mean,
    ) -> ProbingResults:
        """Build a pandas DataFrame with stored values.

        Parameters
        ----------
        aggregate_by : t.Sequence[str] or None, default=None
            If given, aggregate results by the provided keys. It must be a sequence of strings
            contained a subset of {'epoch', 'metric_name', 'module', 'batch_index'}. The function
            used to aggregate values that fall in the same bucket is `aggregate_fn`.

        aggregate_fn : callable or sequence of callables, default=numpy.mean
            Function, or sequence of callables used to aggregate values in a same bucket. Also
            accept values supported by pandas Aggregate function. Must receive a sequence of
            values, and return a single number. Ignored if `aggregate_by=None`.

        Returns
        -------
        dataframes : ProbingResults
            Results casted to pandas DataFrames.
        """

        common_kwargs = dict(aggregate_by=aggregate_by, aggregate_fn=aggregate_fn)

        ret: list[pd.DataFrame] = [self.train.to_pandas(**common_kwargs)]

        if self.eval:
            ret.append(self.eval.to_pandas(**common_kwargs))

        if self.test:
            ret.append(self.test.to_pandas(**common_kwargs))

        return ProbingResults(*ret)
