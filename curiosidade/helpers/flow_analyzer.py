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

    module_refs: list[tuple[str, torch.nn.Module]] = []
    last_probed_module_to_end: list[t.Optional[str]] = [None]

    def hook_pre_fn(module: torch.nn.Module, *args: t.Any, **kwargs: t.Any) -> None:
        # A module started (probed or not)
        # Should it be ignored?
        #  - If it is a probed module, it will eventually ends and clear everything, so no
        #    need to worry anyway.
        #  - If it is not a probed module:
        #       - If it is within a probed module, it will end and clear this module. Ok.
        #       - If it is not within a probled module, good candidate to prune. Ok.
        # Ok??
        module_name = kwargs["module_name"]
        module_refs.append((module_name, module))

    def hook_post_fn(module: torch.nn.Module, *args: t.Any, **kwargs: t.Any) -> None:
        # Probed module ended now, therefore nothing can be ignored up to this point.
        module_name = kwargs["module_name"]
        last_probed_module_to_end[0] = module_name
        module_refs.clear()

    for module_name, module in base_model.named_modules():
        fn_module = functools.partial(hook_pre_fn, module_name=module_name)
        pre_hooks.append(module.register_forward_pre_hook(fn_module))

        if module_name in probed_modules:
            fn_module = functools.partial(hook_post_fn, module_name=module_name)
            post_hooks.append(module.register_forward_hook(fn_module))

    try:
        yield (module_refs, last_probed_module_to_end)

    finally:
        while pre_hooks:
            pre_hooks.pop().remove()

        while post_hooks:
            post_hooks.pop().remove()


def find_unnecessary_modules(
    sample_batch: t.Any,
    base_model: adapters_base.BaseAdapter,
    probed_modules: t.Collection[torch.nn.Module],
    repeat: int = 1,
) -> tuple[tuple[str, torch.nn.Module], ...]:
    base_model.to("cpu")
    base_model.eval()

    probed_set = set(probed_modules)
    unnecessary_modules: tuple[tuple[str, torch.nn.Module], ...] = tuple()
    last_probed: t.Optional[str] = None

    with torch.no_grad(), analyze_modules(base_model, probed_set) as (prune_cand, last_probed_cand):
        for _ in range(repeat):
            input_feats, _ = base_model.break_batch(sample_batch)
            base_model(input_feats)
            temp = tuple(prune_cand)

            non_deterministic_behaviour = unnecessary_modules and (
                temp != unnecessary_modules or last_probed != last_probed_cand[0]
            )

            if non_deterministic_behaviour:
                warnings.warn(
                    message=(
                        "Non-deterministic behaviour detected while inferring unnecessary modules "
                        "for prober training. Will not prune any module, and the full model will be "
                        "loaded on the chosen device."
                    ),
                    category=UserWarning,
                )
                return tuple()

            unnecessary_modules = temp
            last_probed = last_probed_cand[0]

    return unnecessary_modules
