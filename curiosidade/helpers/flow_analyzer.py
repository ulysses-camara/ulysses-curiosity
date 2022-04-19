import typing as t
import contextlib
import warnings
import functools

import torch
import regex

from ..adapters import base as adapters_base


__all__ = [
    "find_unnecessary_modules",
]


@contextlib.contextmanager
def analyze_modules(
    base_model: adapters_base.BaseAdapter, probed_modules: t.Set[torch.nn.Module]
) -> t.Iterator[list[torch.nn.Module]]:
    pre_hooks: list[torch.utils.hooks.RemovableHandle] = []
    post_hooks: list[torch.utils.hooks.RemovableHandle] = []

    unnecessary_cand: list[tuple[str, torch.nn.Module]] = []

    # Note: it is necessary to keep a collection of 'modules that can not be ignored' since
    # modules are reused.
    cant_be_ignored: set[tuple[str, torch.nn.Module]] = {""}

    def hook_pre_fn(module: torch.nn.Module, *args: t.Any, **kwargs: t.Any) -> None:
        # A module started its forward (may be probed or not)
        # Should it be ignored?
        #  - If it is a probed module, it will eventually ends and clear everything, so no
        #    need to worry anyway.
        #  - If it is not a probed module:
        #       - If it is within a probed module, it will end and clear this module. Ok.
        #       - If it is not within a probled module, good candidate to prune. Ok.
        # Ok.
        module_name = kwargs["module_name"]
        unnecessary_cand.append((module_name, module))

    def hook_post_fn(module: torch.nn.Module, *args: t.Any, **kwargs: t.Any) -> None:
        # Probed module ended now, therefore nothing can be ignored up to this point.
        module_name = kwargs["module_name"]
        cant_be_ignored.update(unnecessary_cand)
        unnecessary_cand.clear()

    for module_name, module in base_model.named_modules():
        fn_module = functools.partial(hook_pre_fn, module_name=module_name)
        pre_hooks.append(module.register_forward_pre_hook(fn_module))

        if module_name in probed_modules:
            fn_module = functools.partial(hook_post_fn, module_name=module_name)
            post_hooks.append(module.register_forward_hook(fn_module))

    try:
        yield (unnecessary_cand, cant_be_ignored)

    finally:
        while pre_hooks:
            pre_hooks.pop().remove()

        while post_hooks:
            post_hooks.pop().remove()


def find_unnecessary_modules(
    sample_batches: t.Sequence[t.Any],
    base_model: adapters_base.BaseAdapter,
    probed_modules: t.Collection[torch.nn.Module],
) -> tuple[tuple[str, torch.nn.Module], ...]:
    base_model.to("cpu")
    base_model.eval()

    probed_set = set(probed_modules)
    unnecessary_modules: tuple[tuple[str, torch.nn.Module], ...] = tuple()

    with torch.no_grad(), analyze_modules(base_model, probed_set) as (prune_cand, cant_be_ignored):
        for sample_batch in sample_batches:
            input_feats, _ = base_model.break_batch(sample_batch)
            base_model(input_feats)
            temp = tuple(item for item in prune_cand if item and item not in cant_be_ignored)

            non_deterministic_behaviour_flag = unnecessary_modules and temp != unnecessary_modules

            if non_deterministic_behaviour_flag:
                warnings.warn(
                    message=(
                        "Non-deterministic behaviour detected while inferring unnecessary modules "
                        "for prober training. Will not prune any module, and the full model will be "
                        "loaded in the chosen device."
                    ),
                    category=UserWarning,
                )
                return tuple()

            unnecessary_modules = temp

    return unnecessary_modules
