"""Handle general user input appropriately."""
import typing as t
import re

import regex
import torch.nn


ModuleInputDimType = t.Optional[t.Union[dict[str, int], t.Sequence[int]]]


def get_fn_select_modules_to_probe(
    modules_to_attach: t.Union[str, t.Pattern[str], t.Sequence[str]]
) -> t.Callable[[str], bool]:
    """Return a boolean function that checks if a given module should be probed."""
    if not isinstance(modules_to_attach, (str, regex.Pattern, re.Pattern)):
        modules_to_attach_set = frozenset(modules_to_attach)
        return lambda module_name: module_name in modules_to_attach_set

    if isinstance(modules_to_attach, str):
        modules_to_attach = r"\s*" + modules_to_attach + r"\s*$"
        modules_to_attach = regex.compile(modules_to_attach)

    compiled_regex: t.Pattern[str] = modules_to_attach

    return lambda module_name: compiled_regex.match(module_name) is not None


def get_probing_model_input_dim(
    modules_input_dim: ModuleInputDimType, default_dim: int, module_name: str, module_index: int
) -> int:
    """Return module input dimension."""
    if modules_input_dim is None:
        return default_dim

    if isinstance(modules_input_dim, dict):
        return modules_input_dim.get(module_name, default_dim)

    return modules_input_dim[module_index]


def find_modules_to_prune(
    module_names_to_prune: t.Collection[str],
    named_modules: t.Sequence[tuple[str, torch.nn.Module]],
    probed_module_names: t.Collection[str],
) -> dict[str, torch.nn.Module]:
    """Collect modules that must be pruned during inference.

    Parameters
    ----------
    module_names_to_prune : t.Collection[str]
        Collection with module names to prune.

    named_modules : t.Sequence[tuple[str, torch.nn.Module]]
        All pairs of pretrained module names and its references.

    probed_module_names : t.Collection[str]
        All pretrained modules that will be probed.

    Returns
    -------
    modules_to_prune : dict[str, torch.nn.Module]
        Dictionary mapping module to prune names to its reference.
    """
    if not module_names_to_prune or not probed_module_names or not named_modules:
        return {}

    modules_to_prune: dict[str, torch.nn.Module] = {}
    module_names_to_prune = set(module_names_to_prune)

    for module_name, module in named_modules:
        if module_name in module_names_to_prune:
            modules_to_prune[module_name] = module

    return modules_to_prune
