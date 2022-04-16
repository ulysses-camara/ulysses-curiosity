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


def flatten_results(res: dict[t.Any, dict[t.Any, list[t.Any]]]) -> dict[t.Any, list[t.Any]]:
    """Merge results from all training epochs into a single list per module."""
    flattened_res = collections.defaultdict(list)

    for _, val_dict in res.items():
        for key, vals in val_dict.items():
            flattened_res[key].extend(vals)

    return flattened_res
