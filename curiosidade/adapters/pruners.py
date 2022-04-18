import typing as t
import contextlib

import torch

from . import base


class PrunedModuleException(RuntimeError):
    pass


@contextlib.contextmanager
def stop_forward_if_pruned(modules: t.Sequence[torch.nn.Module]) -> None:
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def stop_forward_fn(layer, *args, **kwargs):
        raise PrunedModuleException

    for module in modules:
        hooks.append(module.register_forward_hook(stop_forward_fn))

    try:
        yield None

    finally:
        while hooks:
            hooks.pop().remove()


class InferencePruner(base.BaseAdapter):
    def __init__(self, model: t.Any):
        super().__init__(model=model, device=model.device)
        self.pruned_modules: tuple[torch.nn.Module, ...] = tuple()

    def register_pruned_modules(
        self, pruned_modules: t.Union[list[torch.nn.Module], torch.nn.Module]
    ) -> "InferencePruner":
        self.pruned_modules = (
            tuple(pruned_modules) if hasattr(pruned_modules, "__len__") else (pruned_modules,)
        )
        return self

    def break_batch(self, batch: t.Any) -> t.Any:
        return self.model.break_batch(batch)

    def forward(self, X: t.Any) -> t.Any:
        with stop_forward_if_pruned(self.pruned_modules):
            try:
                out = self.model(X)

            except PrunedModuleException:
                out = None

        return out
