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


# pylint: disable='invalid-name'


class HuggingfaceAdapter(base.BaseAdapter):
    """Adapter for Huggingface (`transformers` package) models."""

    @staticmethod
    def break_batch(batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Break batch in inputs `X` and input labels `y` appropriately.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Mapping from model inference argument names to corresponding PyTorch Tensors.
            If is expected that `batch` has the key `labels`, and every other entry in
            `batch` has a matching keyword argument in `model` call.

        Returns
        -------
        X : dict[str, torch.Tensor]
            Input features (batch without `labels`).

        y : torch.Tensor
            Label features.
        """
        y = batch.pop("labels")
        X = batch
        return X, y

    def forward(self, X: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Model forward pass.

        Parameters
        ----------
        X : dict[str, torch.Tensor]
            Input pack for transformer model.

        Returns
        -------
        out : dict[str, torch.Tensor]
            Forward pass output.
        """
        for key, val in X.items():
            X[key] = val.to(self.device)

        out = self.model(**X)

        return out

    def named_modules(self) -> t.Iterator[tuple[str, torch.nn.Module]]:
        """Return Torch module .named_modules() iterator."""
        return self.model.named_modules()


class TorchModuleAdapter(base.BaseAdapter):
    """Adapter for PyTorch (`torch` package) modules (`torch.nn.Module`)."""

    @staticmethod
    def break_batch(batch: t.Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Break batch in inputs `X` and input labels `y` appropriately.

        Parameters
        ----------
        batch : tuple[torch.Tensor, ...]
            Tuple in (X, y, *args) format, where `X` is the input features, `y` is the
            corresponding labels, and *args are ignored (if any).

        Returns
        -------
        X : torch.Tensor
            Input features (batch without `labels`).

        y : torch.Tensor
            Label features.
        """
        X, y, *_ = batch
        return X, y

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Model forward pass.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor for model.

        Returns
        -------
        out : torch.Tensor
            Forward pass output.
        """
        X = X.to(self.device)
        out = self.model(X)
        return out

    def named_modules(self) -> t.Iterator[tuple[str, torch.nn.Module]]:
        """Return Torch module .named_modules() iterator."""
        return self.model.named_modules()


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
