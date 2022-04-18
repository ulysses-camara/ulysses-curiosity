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
    module_names_to_prune: t.Optional[t.Union[t.Sequence[str], t.Literal["infer"]]],
    named_modules: t.Sequence[tuple[str, torch.nn.Module]],
    probed_module_names: t.Collection[str],
) -> tuple[torch.nn.Module, ...]:
    if not module_names_to_prune or not probed_module_names or not named_modules:
        return tuple()

    if module_names_to_prune == "infer":
        last_probed_ind = -1
        probed_module_names = frozenset(probed_module_names)

        for i, (module_name, module) in enumerate(named_modules):
            if module_name in probed_module_names:
                last_probed_ind = i

        if not 0 <= last_probed_ind < len(named_modules) - 1:
            return tuple()

        _, module_to_prune = named_modules[last_probed_ind + 1]
        return (module_to_prune,)

    modules_to_prune: list[torch.nn.Module] = []
    module_names_to_prune = set(module_names_to_prune)

    for module_name, module in named_modules:
        if module_name in module_names_to_prune:
            modules_to_prune.append(module)

    return tuple(modules_to_prune)
