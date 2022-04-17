"""Handle general user input appropriately."""
import typing as t
import re

import regex


ModuleInputDimType = t.Optional[t.Union[dict[str, int], t.Sequence[int]]]


def get_fn_select_modules_to_probe(
    modules_to_attach: t.Union[str, t.Pattern[str], t.Sequence[str]]
) -> t.Callable[[str], bool]:
    """Return a boolean function that checks if a given module should be probed."""
    if isinstance(modules_to_attach, (str, regex.Pattern, re.Pattern)):
        compiled_regex: t.Pattern[str] = (
            regex.compile(modules_to_attach)
            if isinstance(modules_to_attach, str)
            else modules_to_attach
        )

        return lambda module_name: compiled_regex.search(module_name) is not None

    modules_to_attach_set = frozenset(modules_to_attach)
    return lambda module_name: module_name in modules_to_attach_set


def get_probing_model_input_dim(
    modules_input_dim: ModuleInputDimType, default_dim: int, module_name: str, module_index: int
) -> int:
    """Return module input dimension."""
    if modules_input_dim is None:
        return default_dim

    if isinstance(modules_input_dim, dict):
        return modules_input_dim.get(module_name, default_dim)

    return modules_input_dim[module_index]
