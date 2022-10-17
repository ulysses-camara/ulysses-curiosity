"""Model architectures built specifically for tests."""
import typing as t
import collections

import torch
import torch.nn


class TorchFF(torch.nn.Module):
    """Simple feedforward torch model."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.params = torch.nn.Sequential(
            collections.OrderedDict(
                (
                    ("lin1", torch.nn.Linear(input_dim, 25, bias=True)),
                    ("relu1", torch.nn.ReLU(inplace=True)),
                    ("lin2", torch.nn.Linear(25, 35, bias=True)),
                    ("relu2", torch.nn.ReLU(inplace=True)),
                    ("lin3", torch.nn.Linear(35, 20, bias=True)),
                    ("relu3", torch.nn.ReLU(inplace=True)),
                    ("lin4", torch.nn.Linear(20, output_dim)),
                )
            ),
        )

    def forward(self, X):
        return self.params(X).squeeze(-1)


class TorchBifurcationInner(torch.nn.Module):
    """PyTorch module with two outputs (bifurcation module)."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.params_a = torch.nn.Sequential(
            collections.OrderedDict(
                (
                    ("lin1_a", torch.nn.Linear(input_dim, 25, bias=True)),
                    ("relu1_a", torch.nn.ReLU(inplace=True)),
                    ("lin2_a", torch.nn.Linear(25, output_dim, bias=True)),
                    ("relu2_a", torch.nn.ReLU(inplace=True)),
                )
            ),
        )

        self.params_b = torch.nn.Sequential(
            collections.OrderedDict(
                (
                    ("lin1_b", torch.nn.Linear(input_dim, 25, bias=True)),
                    ("relu1_b", torch.nn.ReLU(inplace=True)),
                    ("lin2_b", torch.nn.Linear(25, output_dim, bias=True)),
                    ("relu2_b", torch.nn.ReLU(inplace=True)),
                )
            ),
        )

    def forward(self, X):
        out_a = self.params_a(X).squeeze(-1)
        out_b = self.params_b(X).squeeze(-1)
        return out_a, out_b


class TorchBifurcationOuter(torch.nn.Module):
    """PyTorch module that has a bifurcation module within."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.params_b = torch.nn.Linear(20, output_dim)

        self.params_a = torch.nn.Sequential(
            collections.OrderedDict(
                (
                    ("lin1", torch.nn.Linear(input_dim, 15, bias=True)),
                    ("relu1", torch.nn.ReLU(inplace=True)),
                    ("lin2", torch.nn.Linear(15, 25, bias=True)),
                    ("bifurcation", TorchBifurcationInner(25, 10)),
                )
            ),
        )

    def forward(self, X):
        out_a, out_b = self.params_a(X)
        out = torch.cat((out_a, out_b), dim=-1)
        out = self.params_b(out)
        return out


class ProbingModelBifurcation(torch.nn.Module):
    """Probing model adapted to bifurcation modules (two outputs)."""

    def __init__(self, input_dim_a: int, input_dim_b: int, output_dim: int):
        super().__init__()

        self.params = torch.nn.Sequential(
            collections.OrderedDict(
                (
                    ("prob_lin1", torch.nn.Linear(input_dim_a + input_dim_b, 25)),
                    ("prob_relu1", torch.nn.ReLU(inplace=True)),
                    ("prob_lin2", torch.nn.Linear(25, output_dim)),
                )
            )
        )

    def forward(self, X_a, X_b):
        out = torch.cat((X_a, X_b), dim=-1)
        out = self.params(out)
        return out


class TorchLSTM(torch.nn.Module):
    """Simple PyTorch LSTM model."""

    def __init__(
        self,
        output_dim: int,
        vocab_size: int = 32,
        embedding_dim: int = 16,
        hidden_dim: int = 32,
        bidirectional: bool = False,
        num_layers: int = 1,
    ):
        super().__init__()

        self._vocab_size = int(vocab_size)

        self.embeddings = torch.nn.Embedding(
            num_embeddings=self._vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        self.lstm = torch.nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.lin_out = torch.nn.Linear((1 + int(bidirectional)) * hidden_dim, output_dim)

    @property
    def vocab_size(self):
        return self._vocab_size

    def forward(self, input_ids: torch.Tensor):
        out = input_ids

        out = self.embeddings(out)
        out, *_ = self.lstm(out)
        out = self.lin_out(out)

        return out


AVAILABLE_MODELS: t.Final[dict[str, torch.nn.Module]] = {
    "torch_ff.pt": TorchFF,
    "torch_bifurcation.pt": TorchBifurcationOuter,
    "torch_lstm.pt": TorchLSTM,
}
