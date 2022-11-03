"""Load and save utils for cached test pretrained models."""
import functools
import pickle
import os

import torch.nn
import numpy as np

import curiosidade


PRETRAINED_MODEL_DIR = os.path.join(os.path.dirname(__file__), "pretrained_models_for_tests_dir")


class SwitchableProber(curiosidade.probers.utils.ProbingModelFeedforward):
    """Prober adapted for either batches of embeded tokens (N, L, E) or embeded sequences (N, E)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layer_dims,
        dropout: float = 0.0,
        include_batch_norm: bool = False,
        pooling_axis: int = 1,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layer_dims=[hidden_layer_dims]
            if np.isscalar(hidden_layer_dims)
            else hidden_layer_dims,
            dropout=dropout,
            include_batch_norm=include_batch_norm,
        )

        self.pooling_fn = functools.partial(torch.mean, axis=pooling_axis)

    def forward(self, token_embeddings=None, sentence_embedding=None, *args, **kwargs):
        # pylint: disable='unused-argument,keyword-arg-before-vararg'
        if sentence_embedding is not None:
            out = sentence_embedding
        else:
            out = token_embeddings

        if out.ndim >= 3:
            out = self.pooling_fn(out)

        out = self.params(out)
        out = out.squeeze(-1)

        return out


def load_pickled_model(model_name: str) -> torch.nn.Module:
    model_uri = os.path.join(PRETRAINED_MODEL_DIR, model_name)

    with open(model_uri, "rb") as f_in_b:
        model = pickle.load(f_in_b)

    return model


def pickle_model(model: torch.nn.Module, model_name: str) -> None:
    os.makedirs(PRETRAINED_MODEL_DIR, exist_ok=True)
    model_uri = os.path.join(PRETRAINED_MODEL_DIR, model_name)

    with open(model_uri, "wb") as f_out_b:
        pickle.dump(model, f_out_b, protocol=pickle.HIGHEST_PROTOCOL)
