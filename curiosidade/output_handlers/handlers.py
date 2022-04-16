"""Handle probing results appropriately."""
import typing as t

import pandas as pd
import numpy as np


class MetricPack:
    def __init__(self):
        self.stored_values: dict[tuple[t.Any, ...], float] = {}

    def __getitem__(self, key) -> float:
        return self.stored_values[key]

    def __iter__(self) -> t.Iterator[tuple[tuple[t.Any, ...], float]]:
        return iter(self.stored_values.items())

    def __repr__(self) -> str:
        pieces: list[str] = [f"MetricPack with {len(self.stored_values)} values stored in:"]
        for i, (key, val) in enumerate(self.stored_values.items()):
            if i > 3:
                break

            pieces.append(f"  {key}: {val}")

        if len(self.stored_values) > 3:
            pieces.append("  ...")

        return "\n".join(pieces)

    def append(
        self, metrics: dict[t.Any, float], *args: t.Any, merge_keys: bool = True
    ) -> "MetricPack":
        for key, val in metrics.items():
            do_merge = merge_keys and hasattr(key, "__len__") and not isinstance(key, str)
            new_key = (*key, *args) if do_merge else (key, *args)
            self.stored_values[new_key] = val

        return self

    def expand_key_dim(self, *args: t.Any) -> "MetricPack":
        new_stored_values: dict[tuple[t.Any, ...], float] = {}

        for key, val in self.stored_values.items():
            new_key = (*args, *key) if not isinstance(key, str) else (*args, key)
            new_stored_values[new_key] = val

        self.stored_values = new_stored_values

        return self

    def combine(self, metrics: "MetricPack") -> "MetricPack":
        self.stored_values.update(metrics.stored_values)
        return self

    def to_pandas(
        self,
        aggregate_by: t.Optional[t.Sequence[str]] = None,
        aggregate_fn: t.Callable[[t.Sequence[float]], float] = np.mean,
    ) -> pd.DataFrame:
        df = pd.DataFrame(
            [[*keys, val] for keys, val in self.stored_values.items()],
            columns=["epoch", "metric_name", "module", "batch_index", "metric"],
        )

        if aggregate_by is not None:
            aggregate_by_set = set(aggregate_by)

            if "metric" in aggregate_by_set:
                raise ValueError("Can not aggregate by 'metric'.")

            if not aggregate_by_set.issubset(df.columns):
                df_columns_set = set(df.columns)
                df_columns_set.remove("metric")

                unknown_cols = tuple(sorted(aggregate_by_set - df_columns_set))

                raise ValueError(
                    f"Unknown aggregation columns: {unknown_cols}. Must be a combination of the "
                    f"following: {sorted(tuple(df_columns_set))}."
                )

            df = df.groupby(aggregate_by).agg(dict(metric=aggregate_fn))
            df.reset_index(inplace=True)

        return df


class ProbingResults(t.NamedTuple):
    """Class to store probing results per split."""

    train: MetricPack
    eval: t.Optional[MetricPack] = None
    test: t.Optional[MetricPack] = None

    def _format_split(self, split_name: str) -> str:
        split = getattr(self, split_name)
        prefix = f"{split_name:}"
        prefix = f"    {prefix:<5} = "
        return f"{prefix}{str(split).replace('  ', '      ')},"

    def __repr__(self) -> str:
        pieces: list[str] = []

        pieces.append(f"{self.__class__.__name__}(")
        pieces.append(self._format_split("train"))
        pieces.append(self._format_split("eval"))
        pieces.append(self._format_split("test"))
        pieces.append(")")

        return "\n".join(pieces)
