"""Interfaces for inference with distinct model APIs."""
from __future__ import annotations
import typing as t

import torch
import torch.nn
import numpy as np

from . import base


IS_TRANSFORMERS_AVAILABLE: bool
IS_SENTENCE_TRANSFORMERS_AVAILABLE: bool

try:
    import transformers

    IS_TRANSFORMERS_AVAILABLE = True

except ImportError:
    IS_TRANSFORMERS_AVAILABLE = False

try:
    import sentence_transformers

    IS_SENTENCE_TRANSFORMERS_AVAILABLE = True

except ImportError:
    IS_SENTENCE_TRANSFORMERS_AVAILABLE = False


class _HuggingfaceDeviceHandler:
    @classmethod
    def _move_batch_to_device(
        cls,
        batch: t.Union[transformers.BatchEncoding, dict[str, t.Any]],
        device: t.Union[torch.device, str],
    ) -> t.Union[transformers.BatchEncoding, dict[str, torch.Tensor]]:
        """Move a transformers batch to device appropriately."""
        try:
            if isinstance(batch, transformers.BatchEncoding):
                batch = batch.to(device)

            else:
                for key, val in batch.items():
                    batch[key] = val.to(device)

        except AttributeError as err:
            raise TypeError(
                f"Input is not a torch.Tensor (got {type(val)}). Maybe you forgot to cast "
                "your dataset to tensors after text tokenization?"
            ) from err

        return batch


class HuggingfaceAdapter(base.BaseAdapter, _HuggingfaceDeviceHandler):
    """Adapter for Huggingface (`transformers` package) models."""

    @classmethod
    def break_batch(
        cls, batch: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Break batch in inputs `input_feats` and input labels `input_labels` appropriately.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Mapping from model inference argument names to corresponding PyTorch Tensors.
            If is expected that `batch` has the key `labels`, and every other entry in
            `batch` has a matching keyword argument in `model` call.

        Returns
        -------
        input_feats : dict[str, torch.Tensor]
            Input features (batch without `labels`).

        input_labels : torch.Tensor
            Label features.
        """
        input_labels = batch.pop("labels" if "labels" in batch.keys() else "label")
        input_feats = batch
        return input_feats, input_labels

    def forward(
        self, input_feats: t.Union[transformers.BatchEncoding, dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Model forward pass.

        Parameters
        ----------
        input_feats : dict[str, torch.Tensor]
            Input pack for transformer model.

        Returns
        -------
        out : dict[str, torch.Tensor]
            Forward pass output.
        """
        input_feats = self._move_batch_to_device(input_feats, self.device)
        out: dict[str, torch.Tensor] = self.model(**input_feats)

        return out

    def named_modules(self) -> t.Iterator[tuple[str, torch.nn.Module]]:
        """Return Torch module .named_modules() iterator."""
        return self.model.named_modules()


class FullSentenceTransformersAdapter(base.BaseAdapter):
    """Adapter for Full Sentence Transformers (`sentence-transformers` package) models."""

    @classmethod
    def break_batch(cls, batch: tuple[list[str], torch.Tensor]) -> tuple[list[str], torch.Tensor]:
        """Break batch in inputs `input_feats` and input labels `input_labels` appropriately.

        Parameters
        ----------
        batch : tuple[list[str], torch.Tensor]
            Tuple with list of sentences as first argument, and labels in a Tensor.

        Returns
        -------
        input_feats : list[str]
            Input sentences.

        input_labels : torch.Tensor
            Sequence label.
        """
        input_feats, input_labels = batch

        # Note: we sort indices now to prevent SBERT 'encode' to change the batch order.
        # If we don't sort here, then the probers will receive the sentence embeddings
        # in a 'random' order, and the optimization won't converge.
        sorted_inds = np.argsort([-len(item) for item in input_feats]).tolist()

        input_feats = [input_feats[i] for i in sorted_inds]
        input_labels = input_labels[sorted_inds]

        return input_feats, input_labels

    def forward(self, input_feats: list[str]) -> torch.Tensor:
        """Model forward pass.

        Parameters
        ----------
        input_feats : list[str]
            Input pack for sentence transformer model.

        Returns
        -------
        out : torch.Tensor
            Forward pass output.
        """
        out: torch.Tensor = self.model.encode(  # type: ignore
            sentences=input_feats,
            batch_size=len(input_feats),  # Note: we assume that 'input_feats' is a mini-batch.
            show_progress_bar=False,
            convert_to_numpy=False,
            convert_to_tensor=True,
            normalize_embeddings=False,
            device=self.device,
        )

        return torch.as_tensor(out)

    def named_modules(self) -> t.Iterator[tuple[str, torch.nn.Module]]:
        """Return Torch module .named_modules() iterator."""
        return self.model.named_modules()


class SentenceTransformersAdapter(base.BaseAdapter, _HuggingfaceDeviceHandler):
    """Adapter for Sentence Transformers (`sentence-transformers` package) models."""

    @classmethod
    def break_batch(
        cls, batch: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Break batch in inputs `input_feats` and input labels `input_labels` appropriately.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Mapping from model inference argument names to corresponding PyTorch Tensors.
            If is expected that `batch` has the key `labels`, and every other entry in
            `batch` has a matching keyword argument in `model` call.

        Returns
        -------
        input_feats : dict[str, torch.Tensor]
            Input features (batch without `labels`).

        input_labels : torch.Tensor
            Label features.
        """
        input_labels = batch.pop("labels" if "labels" in batch.keys() else "label")
        input_feats = batch
        return input_feats, input_labels

    def forward(
        self, input_feats: t.Union[transformers.BatchEncoding, dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Model forward pass.

        Parameters
        ----------
        input_feats : dict[str, torch.Tensor]
            Input pack for transformer model.

        Returns
        -------
        out : dict[str, torch.Tensor]
            Forward pass output.
        """
        input_feats = self._move_batch_to_device(input_feats, self.device)
        out = self.model(input_feats)["token_embeddings"]
        return torch.as_tensor(out)

    def named_modules(self) -> t.Iterator[tuple[str, torch.nn.Module]]:
        """Return Torch module .named_modules() iterator."""
        return self.model.named_modules()


class TorchModuleAdapter(base.BaseAdapter):
    """Adapter for PyTorch (`torch` package) modules (`torch.nn.Module`)."""

    @classmethod
    def break_batch(cls, batch: t.Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Break batch in inputs `input_feats` and input labels `input_labels` appropriately.

        Parameters
        ----------
        batch : tuple[torch.Tensor, ...]
            Tuple in (input_feats, input_labels, *args) format, where `input_feats` is the input
            features, `input_labels` is the corresponding labels, and *args are ignored (if any).

        Returns
        -------
        input_feats : torch.Tensor
            Input features (batch without `labels`).

        input_labels : torch.Tensor
            Label features.
        """
        input_feats, input_labels, *_ = batch
        return input_feats, input_labels

    def forward(self, input_feats: torch.Tensor) -> torch.Tensor:
        """Model forward pass.

        Parameters
        ----------
        input_feats : torch.Tensor
            Input tensor for model.

        Returns
        -------
        out : torch.Tensor
            Forward pass output.
        """
        input_feats = input_feats.to(self.device)
        out = self.model(input_feats)
        return torch.as_tensor(out)

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

    if IS_SENTENCE_TRANSFORMERS_AVAILABLE and isinstance(
        model, sentence_transformers.SentenceTransformer
    ):
        return FullSentenceTransformersAdapter(model, *args, **kwargs)

    if IS_SENTENCE_TRANSFORMERS_AVAILABLE and isinstance(
        model, sentence_transformers.models.Transformer
    ):
        return SentenceTransformersAdapter(model, *args, **kwargs)

    if isinstance(model, torch.nn.Module):
        return TorchModuleAdapter(model, *args, **kwargs)

    raise TypeError(
        f"Unknown model type '{type(model)}'. Please provide a Huggingface transformer or "
        "a PyTorch module."
    )
