"""Handle general user input appropriately."""
import typing as t

import regex


ModuleInputDimType = t.Optional[t.Union[dict[str, int], t.Sequence[int]]]


def get_fn_select_modules_to_probe(
    modules_to_attach: t.Union[str, regex.Pattern, t.Sequence[str]]
) -> t.Callable[[str], bool]:
    """Return a boolean function that checks if a given module should be probed."""
    if isinstance(modules_to_attach, str):
        modules_to_attach = regex.compile(modules_to_attach)
        return lambda module_name: modules_to_attach.search(module_name) is not None

    modules_to_attach = frozenset(modules_to_attach)
    return lambda module_name: module_name in modules_to_attach


def get_module_input_dim(
    modules_input_dim: ModuleInputDimType, default_dim: int, module_name: str, module_index: int
) -> int:
    """Return module input dimension."""
    if modules_input_dim is None:
        return default_dim

    if isinstance(modules_input_dim, dict):
        return modules_input_dim.get(module_name, default_dim)

    return modules_input_dim[module_index]
