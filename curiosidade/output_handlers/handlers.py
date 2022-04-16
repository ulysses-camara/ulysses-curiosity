"""Handle probing results appropriately."""
import typing as t
import collections


MetricDictType = dict[t.Union[int, str], t.Any]


class ProbingResults(t.NamedTuple):
    """Class to store probing results per split."""

    train: MetricDictType
    eval: t.Optional[MetricDictType] = None
    test: t.Optional[MetricDictType] = None

    def _format_split(self, split_name: str) -> str:
        split = getattr(self, split_name)
        prefix = f"{split_name:}"
        prefix = f"    {prefix:<5} = "

        if split is not None:
            val_len = len(next(iter(split.values())))
            suffix = f"dict with modules {tuple(split.keys())} and {val_len} values each"

        else:
            suffix = str(None)

        return f"{prefix}{suffix},"

    def __repr__(self) -> str:
        pieces: list[str] = []

        pieces.append(f"{self.__class__.__name__}(")
        pieces.append(self._format_split("train"))
        pieces.append(self._format_split("eval"))
        pieces.append(self._format_split("test"))
        pieces.append(")")

        return "\n".join(pieces)


def flatten(res: dict[t.Any, dict[t.Any, list[t.Any]]]) -> dict[t.Any, list[t.Any]]:
    """Merge results from all training epochs into a single list per module."""
    flat_res = collections.defaultdict(list)

    for _, val_dict in res.items():
        for key, vals in val_dict.items():
            if hasattr(vals, "__len__"):
                flat_res[key].extend(vals)
            else:
                flat_res[key].append(vals)

    return flat_res


def aggregate(
    res: dict[t.Any, dict[t.Any, list[t.Any]]], agg_fn: t.Callable[[t.Sequence[float]], float]
) -> dict[t.Any, float]:
    """Aggregate results into a single value, for every epoch, using `agg_fn`."""
    agg_res = collections.defaultdict(dict)

    for key_a, val_dict in res.items():
        for key_b, vals in val_dict.items():
            agg_res[key_a][key_b] = agg_fn(vals)

    return agg_res
