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


def find_modules_to_prune(
    module_names_to_prune: t.Optional[t.Union[t.Collection[str], t.Literal["infer"]]],
    named_modules: t.Sequence[tuple[str, torch.nn.Module]],
    probed_module_names: t.Collection[str],
) -> dict[str, torch.nn.Module]:
    """Collect modules that must be pruned during inference.

    Parameters
    ----------
    modules_names_to_prune : t.Collection[str] or 'infer'
        Collection with module names to prune, or 'infer'. If 'infer', the first non-probed
        module after every probed module is assumed to be the only pruned module. This
        heuristic only works properly in 'one-dimensional' models with deterministic flows.

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

    if module_names_to_prune == "infer":
        last_probed_ind = -1
        probed_module_names = frozenset(probed_module_names)

        for i, (module_name, module) in enumerate(named_modules):
            if module_name in probed_module_names:
                last_probed_ind = i

        if not 0 <= last_probed_ind < len(named_modules) - 1:
            return {}

        named_modules = tuple(named_modules)

        module_name, module_to_prune = named_modules[last_probed_ind + 1]
        return {module_name: module_to_prune}

    modules_to_prune: dict[str, torch.nn.Module] = {}
    module_names_to_prune = set(module_names_to_prune)

    for module_name, module in named_modules:
        if module_name in module_names_to_prune:
            modules_to_prune[module_name] = module

    return modules_to_prune
