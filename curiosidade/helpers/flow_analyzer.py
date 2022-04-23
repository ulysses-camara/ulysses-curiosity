"""Gather information about pretrained model by forwarding sample batches."""
import typing as t
import contextlib
import warnings
import functools

import torch

from ..adapters import base as adapters_base


__all__ = [
    "run_inspection_batches",
]


TensorType = t.Union[torch.Tensor, tuple[torch.Tensor, ...]]


class _AnalyzerContainer:
    """Container for information gathered while exploring pretrained model."""

    def __init__(self) -> None:
        self.unnecessary_cand: list[tuple[str, torch.nn.Module]] = []
        self.probing_input_dims: dict[str, tuple[int, ...]] = {}

        # Note: it is necessary to keep a collection of 'modules that can not be ignored' since
        # modules are reused.
        self.cant_be_ignored: set[tuple[str, torch.nn.Module]] = set()

    def dismiss_unnecessary_cand(self) -> "_AnalyzerContainer":
        """Clear current unnecessary module candidates, and mark then as `can't be ignored`."""
        self.cant_be_ignored.update(self.unnecessary_cand)
        self.unnecessary_cand.clear()
        return self

    def register_output_shape(self, module_name: str, m_output: TensorType) -> "_AnalyzerContainer":
        """Register output shape for the given module."""
        if torch.is_tensor(m_output):
            m_output = (m_output,)  # type: ignore

        out_shapes: tuple[int, ...] = tuple(item.shape[-1] for item in m_output)
        self.probing_input_dims[module_name] = out_shapes

        return self


@contextlib.contextmanager
def analyze_modules(
    base_model: adapters_base.BaseAdapter, probed_modules: t.Set[str]
) -> t.Iterator[_AnalyzerContainer]:
    """Insert temporary hooks in pretrained model to collect information."""
    pre_hooks: list[torch.utils.hooks.RemovableHandle] = []
    post_hooks: list[torch.utils.hooks.RemovableHandle] = []

    channel_container = _AnalyzerContainer()
    channel_container.cant_be_ignored.add(("", base_model.get_torch_module()))

    def hook_pre_fn(
        module: torch.nn.Module, *args: t.Any, module_name: str, **kwargs: t.Any
    ) -> None:
        # pylint: disable='unused-argument'
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
        channel_container.dismiss_unnecessary_cand()
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
    probed_modules: t.Collection[str],
) -> dict[str, t.Any]:
    """Gather information about pretrained model by forwaring sample batches.

    Function used to infer probing input dimensions, and also detect pretrained modules
    unnecessary to train probing models.

    Parameters
    ----------
    sample_batches : t.Sequence[t.Any]
        Sample batches to forward to the pretrained model. Only a single batch should suffice, but
        additional batches may be used to detect any form of non-deterministic behaviour in the
        forward phase of the pretrained model. If this is the case, pretrained modules will be
        deemed unnecessary to train probing models, at the expense of extra computational cost.

    base_model : adapters.base.BaseAdapter
        Properly adapted (or even extended) pretrained model to probe.

    probed_modules : t.Collection[str]
        Names of all modules that will be probed.

    Returns
    -------
    inspection_results : dict[str, t.Any]
        Dictionary containing information about pretrained model architecture, containing the
        following keys:
        - `probing_input_dims`: dictionary mapping probed modules to its input dimensions (tuples).
        - `unnecessary_modules`: tuple containing all modules deemed unnecessary for probing model
          training, following the (module_name, module_reference) pair format.
    """
    base_model.eval()

    unnecessary_modules: tuple[tuple[str, torch.nn.Module], ...] = tuple()
    probing_input_dims: dict[str, tuple[int, ...]] = {}

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
                        "for prober training. Will not prune any module, and the full model will "
                        "be loaded in the chosen device."
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
