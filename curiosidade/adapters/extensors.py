"""Extensors expand the API of model Adapters."""
import typing as t
import contextlib

import torch

from . import base


class PrunedModuleException(RuntimeError):
    """Exception raised when calling forward in a pruned module."""


@contextlib.contextmanager
def stop_forward_if_pruned(modules: t.Sequence[torch.nn.Module]) -> t.Iterator[None]:
    """Raise PrunedModuleException whenever `modules` forward are called."""
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


class InferencePrunerExtensor(base.BaseExtensor):
    """Extend adapter API with module pruning."""

    def __init__(self, model: base.BaseAdapter):
        super().__init__(model=model)
        self._pruned_modules: dict[str, torch.nn.Module] = {}

    def __repr__(self) -> str:
        pieces: list[str] = [f"{self.__class__.__name__}({self.model})"]

        if self.has_pruned_modules:
            pruned_module_names = self.pruned_module_names
            pieces.append(f"(a): Pruned module(s) ({len(pruned_module_names)} in total):")

            for i, pruned_modules_name in enumerate(pruned_module_names):
                pieces.append(f"  ({i}): {pruned_modules_name}")

        else:
            pieces.append("(a): No pruned modules.")

        return "\n".join(pieces)

    def register_pruned_modules(
        self, pruned_modules: dict[str, torch.nn.Module]
    ) -> "InferencePrunerAdapter":
        """Updated in-place pruned modules with all values in `pruned_modules`."""
        self._pruned_modules.update(pruned_modules)
        return self

    @property
    def pruned_module_names(self) -> tuple[str, ...]:
        """Return names of pruned modules."""
        return tuple(self._pruned_modules.keys())

    @property
    def pruned_modules(self) -> tuple[torch.nn.Module, ...]:
        """Return pruned module references."""
        return tuple(self._pruned_modules.values())

    @property
    def has_pruned_modules(self) -> bool:
        """Check whether adapter has pruned modules."""
        return len(self._pruned_modules) > 0

    def clear_pruned_modules(self) -> "InferencePrunerAdapter":
        """Clear in-place all pruned modules."""
        self._pruned_modules.clear()
        return self

    def forward(self, input_feats: t.Any) -> t.Any:
        """Forward `input_feats` stopping at the first pruned module."""
        with stop_forward_if_pruned(self.pruned_modules):
            try:
                out = self.model(input_feats)

            except PrunedModuleException:
                out = None

        return out
