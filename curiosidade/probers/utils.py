"""Utility function related to probing model creation."""
import typing as t
import functools

import torch
import torch.nn
import numpy as np


__all__ = [
    "get_probing_model_feedforward",
    "get_probing_model_for_sequences",
]


ProbingModelType = t.Callable[[int, int], torch.nn.Module]


class ProbingModelFeedforward(torch.nn.Module):
    """Create a simple feedforward probing model.

    The activation functions for hidden layers are Rectified Linear Units (ReLU).

    Each layer of this architecture is as follows:

    1. Linear
    2. BatchNorm (if `include_batch_norm=True`)
    3. ReLU
    4. Dropout (if `dropout > 0.0`)

    Parameters
    ----------
    input_dim : int
        Input dimension, which must correspond to the output dimension of the attached
        pretrained layer.

    output_dim : int
        Output dimension of probing model, which depends on the probing task specification.

    hidden_layer_dims : t.Sequence[int]
        Dimension of hidden layers.

    include_batch_norm : bool, default=False
        If True, include Batch Normalization between Linear and ReLU modules.

    dropout : float, default=0.0
        Amount of dropout per layer.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layer_dims: t.Sequence[int],
        include_batch_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        dims = np.hstack((input_dim, *hidden_layer_dims))

        self.params = torch.nn.Sequential(
            *[
                self._create_layer(
                    input_dim=dims[i],
                    output_dim=dims[i + 1],
                    include_batch_norm=include_batch_norm,
                    dropout=dropout,
                )
                for i in range(len(dims) - 1)
            ],
            torch.nn.Linear(dims[-1], output_dim),
        )

        self.dims = tuple(np.hstack((dims, output_dim)))

    @staticmethod
    def _create_layer(
        input_dim: int, output_dim: int, include_batch_norm: bool, dropout: float
    ) -> torch.nn.Sequential:
        layer: list[torch.nn.Module] = [
            torch.nn.Linear(input_dim, output_dim, bias=not include_batch_norm),
            torch.nn.ReLU(inplace=True),
        ]

        if include_batch_norm:
            layer.insert(1, torch.nn.BatchNorm1d(output_dim))

        if dropout > 0.0:
            layer.append(torch.nn.Dropout(p=dropout))

        return torch.nn.Sequential(*layer)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # pylint: disable='missing-function-docstring', 'invalid-name'
        out = self.params(X)  # type: torch.Tensor
        out = out.squeeze(-1)
        return out


class ProbingModelForSequences(ProbingModelFeedforward):
    """Create a simple feedforward probing model for sequence models.

    The activation functions for hidden layers are Rectified Linear Units (ReLU).

    At the start of the feedforward process, the activations are pooled using a function specified
    by `pooling_strategy` argument, in the token axis (defined by `pooling_axis` argument),
    transforming inputs to fixed-length tensors.

    Parameters
    ----------
    input_dim : int
        Input dimension, which must correspond to the output dimension of the attached
        pretrained layer.

    output_dim : int
        Output dimension of probing model, which depends on the probing task specification.

    hidden_layer_dims : t.Sequence[int]
        Dimension of hidden layers.

    pooling_strategy : {'max', 'mean'}, default='max'
        Pooling strategy, to transform variable-length tensors into fixed-length tensors.

    pooling_axis : int, default=1
        Axis to apply pooling.

    include_batch_norm : bool, default=False
        If True, include Batch Normalization between Linear and ReLU modules.

    dropout : float, default=0.0
        Amount of dropout per layer.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layer_dims: t.Sequence[int],
        pooling_strategy: t.Literal["max", "mean"] = "max",
        pooling_axis: int = 1,
        include_batch_norm: bool = False,
        dropout: float = 0.0,
    ):
        if pooling_strategy not in {"max", "mean"}:
            raise ValueError(
                f"Pooling strategy must be either 'max' ou 'mean' (got {pooling_strategy})."
            )

        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layer_dims=hidden_layer_dims,
            include_batch_norm=include_batch_norm,
            dropout=dropout,
        )

        if pooling_strategy == "max":

            def pooling_fn(inp: torch.Tensor) -> torch.Tensor:
                out: torch.Tensor
                out, _ = inp.max(dim=pooling_axis)
                return out

            self.pooling_fn = pooling_fn

        else:
            self.pooling_fn = functools.partial(torch.mean, axis=pooling_axis)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        out = args[0]  # shape: (batch_size, max_sequence_length, embed_dim)
        out = self.pooling_fn(out)  # shape: (batch_size, embed_dim)
        out = self.params(out)  # shape: (batch_size, output_dim)
        out = out.squeeze(-1)  # shape: (batch_size, output_dim?)
        return out


def get_probing_model_feedforward(
    hidden_layer_dims: t.Sequence[int], include_batch_norm: bool = False, dropout: float = 0.0
) -> ProbingModelType:
    """Get a ``ProbingModelFeedforward`` architecture.

    Parameters
    ----------
    hidden_layer_dims : t.Sequence[int]
        Dimension of hidden layers.

    include_batch_norm : bool, default=False
        If True, include Batch Normalization between Linear and ReLU modules.

    dropout : float, default=0.0
        Amount of dropout per layer.

    Returns
    -------
    architecture : t.Callable[[int, ...], ProbingModelFeedforward]
        Callable that generates the corresponding probing model.
    """
    return functools.partial(
        ProbingModelFeedforward,
        hidden_layer_dims=hidden_layer_dims,
        include_batch_norm=include_batch_norm,
        dropout=dropout,
    )


def get_probing_model_for_sequences(
    hidden_layer_dims: t.Sequence[int],
    pooling_strategy: t.Literal["max", "mean"] = "max",
    pooling_axis: int = 1,
    include_batch_norm: bool = False,
    dropout: float = 0.0,
) -> ProbingModelType:
    """Get a ``ProbingModelForSequences`` architecture.

    This probing model architecture handles variable-length inputs, by applying a pooling
    function in a variable-length axis, therefore transforming the inputs to fixed-length
    representations.

    Parameters
    ----------
    hidden_layer_dims : t.Sequence[int]
        Dimension of hidden layers.

    pooling_strategy : {'max', 'mean'}, default='max'
        Pooling strategy, to transform variable-length tensors into fixed-length tensors.

    pooling_axis : int, default=1
        Axis to apply pooling.

    include_batch_norm : bool, default=False
        If True, include Batch Normalization between Linear and ReLU modules.

    dropout : float, default=0.0
        Amount of dropout per layer.

    Returns
    -------
    architecture : t.Callable[[int, ...], ProbingModelForSequences]
        Callable that generates the corresponding probing model.
    """
    return functools.partial(
        ProbingModelForSequences,
        hidden_layer_dims=hidden_layer_dims,
        pooling_strategy=pooling_strategy,
        pooling_axis=pooling_axis,
        include_batch_norm=include_batch_norm,
        dropout=dropout,
    )
