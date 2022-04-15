"""Interfaces for inference with distinct model APIs."""
import typing as t

import torch
import torch.nn

from . import base


IS_TRANSFORMERS_AVAILABLE: t.Final[bool]

try:
    import transformers

    IS_TRANSFORMERS_AVAILABLE = True

except ImportError:
    IS_TRANSFORMERS_AVAILABLE = False


class HuggingfaceAdapter(base.BaseAdapter):
    """Adapter for Huggingface (`transformers` package) models."""

    def forward(self, batch: dict[str, torch.Tensor]) -> base.AdapterInferenceOutputType:
        y = batch.pop("labels")

        for key, val in batch.items():
            batch[key] = val.to(self.device)

        X = batch
        out = self.model(**X)

        return out, X, y


class TorchModuleAdapter(base.BaseAdapter):
    """Adapter for PyTorch (`torch` package) modules (`torch.nn.Module`)."""

    def forward(
        self, batch: tuple[torch.Tensor, torch.Tensor, ...]
    ) -> base.AdapterInferenceOutputType:
        """Inference with"""
        X, y, *_ = batch
        X = X.to(self.device)
        out = self.model(X)
        return out, X, y


def get_model_adapter(model: t.Any, *args: t.Any, **kwargs: t.Any) -> base.BaseAdapter:
    """Factory function to deduce a model appropriate inference adapter."""
    if IS_TRANSFORMERS_AVAILABLE and isinstance(model, transformers.PreTrainedModel):
        return HuggingfaceAdapter(model, *args, **kwargs)

    if isinstance(model, torch.nn.Module):
        return TorchModuleAdapter(model, *args, **kwargs)

    raise TypeError(
        f"Unknown model type '{type(model)}'. Please provide a Huggingface transformer or "
        "a PyTorch module."
    )
