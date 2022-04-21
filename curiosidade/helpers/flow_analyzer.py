import typing as t
import contextlib
import warnings
import functools

import torch
import regex

from ..adapters import base as adapters_base


__all__ = [
    "run_inspection_batches",
]


TensorType = t.Union[torch.Tensor, tuple[torch.Tensor, ...]]


class _AnalyzerContainer(t.NamedTuple):
    unnecessary_cand: list[tuple[str, torch.nn.Module]] = []
    probing_input_dims: dict[str, list[tuple[int, ...]]] = {}

    # Note: it is necessary to keep a collection of 'modules that can not be ignored' since
    # modules are reused.
    cant_be_ignored: set[tuple[str, torch.nn.Module]] = {""}

    def update_unnecessary_cand(self) -> "_AnalyzerContainer":
        self.cant_be_ignored.update(self.unnecessary_cand)
        self.unnecessary_cand.clear()
        return self

    def register_output_shape(self, module_name: str, m_output: TensorType) -> "_AnalyzerContainer":
        if torch.is_tensor(m_output):
            m_output = (m_output,)

        out_shapes = tuple(item.shape[-1] for item in m_output)
        self.probing_input_dims[module_name] = out_shapes

        return self


@contextlib.contextmanager
def analyze_modules(
    base_model: adapters_base.BaseAdapter, probed_modules: t.Set[torch.nn.Module]
) -> t.Iterator[_AnalyzerContainer]:
    pre_hooks: list[torch.utils.hooks.RemovableHandle] = []
    post_hooks: list[torch.utils.hooks.RemovableHandle] = []

    channel_container = _AnalyzerContainer()

    def hook_pre_fn(
        module: torch.nn.Module, *args: t.Any, module_name: str, **kwargs: t.Any
    ) -> None:
        # A module started its forward (may be probed or not)
        # Should it be ignored?
        #  - If it is a probed module, it will eventually ends and clear everything, so no
        #    need to worry anyway.
        #  - If it is not a probed module:
        #       - If it is within a probed module, it will end and clear this module. Ok.
        #       - If it is not within a probled module, good candidate to prune. Ok.
        # Ok.
        channel_container.unnecessary_cand.append((module_name, module))

    def hook_post_fn(
        module: torch.nn.Module,
        m_input: TensorType,
        m_output: TensorType,
        module_name: str,
        **kwargs: t.Any
    ) -> None:
        # pylint: disable='unused-argument'
        # Probed module ended now, therefore nothing can be ignored up to this point.
        channel_container.update_unnecessary_cand()
        channel_container.register_output_shape(module_name, m_output)

    for module_name, module in base_model.named_modules():
        fn_module = functools.partial(hook_pre_fn, module_name=module_name)
        pre_hooks.append(module.register_forward_pre_hook(fn_module))

        if module_name in probed_modules:
            fn_module = functools.partial(hook_post_fn, module_name=module_name)
            post_hooks.append(module.register_forward_hook(fn_module))

    try:
        yield channel_container

    finally:
        while pre_hooks:
            pre_hooks.pop().remove()

        while post_hooks:
            post_hooks.pop().remove()


def run_inspection_batches(
    sample_batches: t.Sequence[t.Any],
    base_model: adapters_base.BaseAdapter,
    probed_modules: t.Collection[torch.nn.Module],
) -> dict[str, t.Any]:
    base_model.eval()

    unnecessary_modules: tuple[tuple[str, torch.nn.Module], ...] = tuple()
    probing_input_dims = dict[str, tuple[int, ...]]

    probed_set = set(probed_modules)

    if torch.is_tensor(sample_batches):
        sample_batches = [sample_batches]

    with torch.no_grad(), analyze_modules(base_model, probed_set) as channel_container:
        for sample_batch in sample_batches:
            input_feats, _ = base_model.break_batch(sample_batch)
            base_model(input_feats)
            temp = tuple(
                item
                for item in channel_container.unnecessary_cand
                if item and item not in channel_container.cant_be_ignored
            )

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
                unnecessary_modules = tuple()
                break

            unnecessary_modules = temp
            probing_input_dims = channel_container.probing_input_dims

    out = dict(
        probing_input_dims=probing_input_dims,
        unnecessary_modules=unnecessary_modules,
    )

    return out
