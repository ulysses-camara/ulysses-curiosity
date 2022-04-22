"""Train test models to simulate pretrained models."""
import typing as t
import collections

import torch
import torch.nn

from . import model_utils


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


AVAILABLE_MODELS: t.Final[dict[str, torch.nn.Module]] = {
    "torch_ff.pt": TorchFF,
}


def gen_random_dataset(
    train_size: int = 250,
    eval_size: int = 50,
    test_size: int = 50,
    num_dim: int = 3,
    num_labels: int = 4,
    random_seed: int = 16,
) -> tuple[torch.utils.data.Dataset[tuple[torch.Tensor, ...]], ...]:
    with torch.random.fork_rng():
        torch.random.manual_seed(random_seed)

        X_train = torch.randn(train_size, num_dim)
        X_eval = torch.randn(eval_size, num_dim)
        X_test = torch.randn(test_size, num_dim)

    y_train = X_train.sum(axis=-1)
    y_eval = X_eval.sum(axis=-1)
    y_test = X_test.sum(axis=-1)

    y_max = y_train.max().item() - 1e-8
    y_min = y_train.min().item()

    def sum_to_labels(y: torch.Tensor) -> torch.Tensor:
        y = ((y - y_min) / (y_max - y_min) * num_labels).floor().long()
        torch.maximum(y, torch.full_like(y, num_labels), out=y)
        torch.minimum(y, torch.zeros_like(y), out=y)
        return y

    y_train = sum_to_labels(y_train)
    y_eval = sum_to_labels(y_eval)
    y_test = sum_to_labels(y_test)

    df_train = torch.utils.data.TensorDataset(X_train, y_train)
    df_eval = torch.utils.data.TensorDataset(X_eval, y_eval)
    df_test = torch.utils.data.TensorDataset(X_test, y_test)

    return df_train, df_eval, df_test, num_labels


def train(
    model_name: str,
    batch_size: int = 16,
    num_train_epochs: int = 100,
    lr: float = 0.01,
    save: bool = True,
):
    df_train, *_ = gen_random_dataset()

    model = AVAILABLE_MODELS[model_name](input_dim=3, output_dim=4)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    dataloader_train = torch.utils.data.DataLoader(df_train, batch_size=batch_size, shuffle=True)

    for _ in range(num_train_epochs):
        for X, y in dataloader_train:
            optim.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optim.step()

    if save:
        model_utils.pickle_model(model=model, model_name=model_name)

    return model
