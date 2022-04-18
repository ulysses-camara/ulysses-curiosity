import typing as t
import contextlib

import torch

from . import base


class PrunedModuleException(RuntimeError):
    pass


@contextlib.contextmanager
def stop_forward_if_pruned(modules: t.Sequence[torch.nn.Module]) -> t.Iterator[None]:
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def stop_forward_fn(*args: t.Any, **kwargs: t.Any) -> None:
        raise PrunedModuleException

    for module in modules:
        hooks.append(module.register_forward_hook(stop_forward_fn))

    try:
        yield None

    finally:
        while hooks:
            hooks.pop().remove()


class InferencePrunerAdapter(base.BaseAdapter):
    def register_pruned_modules(
        self, pruned_modules: dict[str, torch.nn.Module]
    ) -> "InferencePrunerAdapter":
        self._pruned_modules.update(pruned_modules)
        return self

    def reset_pruned_modules(self) -> "InferencePrunerAdapter":
        self._pruned_modules.clear()
        return self

    def break_batch(self, batch: t.Any) -> t.Any:
        return self.model.break_batch(batch)  # type: ignore

    def forward(self, X: t.Any) -> t.Any:
        with stop_forward_if_pruned(self.pruned_modules):
            try:
                out = self.model(X)

            except PrunedModuleException:
                out = None

        return out

    def named_modules(self) -> t.Iterator[tuple[str, torch.nn.Module]]:
        """Return Torch module .named_modules() iterator."""
        return self.model.named_modules()
