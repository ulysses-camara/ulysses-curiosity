"""Train test models to simulate pretrained models."""
import typing as t

import torch
import torch.nn

from . import model_utils
from . import architectures


def gen_random_dataset(
    train_size: int = 250,
    eval_size: int = 50,
    test_size: int = 50,
    num_dim: int = 3,
    num_labels: int = 4,
    random_seed: int = 16,
    integer_data: bool = False,
    vocab_size: int = 128,
) -> tuple[torch.utils.data.Dataset[tuple[torch.Tensor, ...]], ...]:
    with torch.random.fork_rng():
        torch.random.manual_seed(random_seed)

        if integer_data:
            X_train = torch.randint(low=1, high=vocab_size, size=(train_size, num_dim))
            X_eval = torch.randint(low=1, high=vocab_size, size=(eval_size, num_dim))
            X_test = torch.randint(low=1, high=vocab_size, size=(test_size, num_dim))

        else:
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
    integer_data: bool = False,
    **kwargs: t.Any,
):
    df_train, *_ = gen_random_dataset(
        integer_data=integer_data, vocab_size=kwargs.get("vocab_size", -1)
    )

    model = architectures.AVAILABLE_MODELS[model_name](**kwargs)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    dataloader_train = torch.utils.data.DataLoader(df_train, batch_size=batch_size, shuffle=True)

    for _ in range(num_train_epochs):
        for X, y in dataloader_train:
            optim.zero_grad()
            y_pred = model(X)

            if y_pred.ndim == 3:
                y_pred = y_pred[:, -1, :]

            loss = criterion(y_pred, y)
            loss.backward()
            optim.step()

    if save:
        model_utils.pickle_model(model=model, model_name=model_name)

    return model
