"""Interfaces for inference with distinct model APIs."""
import typing as t

import torch
import torch.nn

from . import base


IS_TRANSFORMERS_AVAILABLE: bool

try:
    import transformers

    IS_TRANSFORMERS_AVAILABLE = True

except ImportError:
    IS_TRANSFORMERS_AVAILABLE = False


class HuggingfaceAdapter(base.BaseAdapter):
    """Adapter for Huggingface (`transformers` package) models."""

    def forward(self, batch: dict[str, torch.Tensor]) -> base.AdapterInferenceOutputType:
        """Model forward pass.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Mapping from model inference argument names to corresponding PyTorch Tensors.
            If is expected that `batch` has the key `labels`, and every other entry in
            `batch` has a matching keyword argument in `model` call.

        Returns
        -------
        out : dict[str, torch.Tensor]
            Forward pass output.

        X : dict[str, torch.Tensor]
            Input features.

        y : torch.Tensor
            Label features.
        """
        y = batch.pop("labels")

        for key, val in batch.items():
            batch[key] = val.to(self.device)

        X = batch
        out = self.model(**X)

        return out, X, y


class TorchModuleAdapter(base.BaseAdapter):
    """Adapter for PyTorch (`torch` package) modules (`torch.nn.Module`)."""

    def forward(self, batch: tuple[torch.Tensor, ...]) -> base.AdapterInferenceOutputType:
        """Model forward pass.

        Parameters
        ----------
        batch : tuple[torch.Tensor, ...]
            Tuple in (X, y, *args) format, where `X` is the input features, `y` is the
            corresponding labels, and *args are ignored (if any).

        Returns
        -------
        out : torch.Tensor
            Forward pass output.

        X : torch.Tensor
            Input features.

        y : torch.Tensor
            Label features.
        """
        X, y, *_ = batch
        X = X.to(self.device)
        out = self.model(X)
        return out, X, y


def get_model_adapter(model: t.Any, *args: t.Any, **kwargs: t.Any) -> base.BaseAdapter:
    """Factory function to deduce a model appropriate inference adapter.

    Parameters
    ----------
    model : t.Any
        Model to be wrapped.

    *args : tuple
        Extra positional arguments to adapter.

    **kwargs : dict
        Extra keywords arguments to adapter.

    Raises
    ------
    TypeError
        If `model` type is not supported.
    """
    if IS_TRANSFORMERS_AVAILABLE and isinstance(model, transformers.PreTrainedModel):
        return HuggingfaceAdapter(model, *args, **kwargs)

    if isinstance(model, torch.nn.Module):
        return TorchModuleAdapter(model, *args, **kwargs)

    raise TypeError(
        f"Unknown model type '{type(model)}'. Please provide a Huggingface transformer or "
        "a PyTorch module."
    )
