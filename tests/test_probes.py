"""Test main features from package (probing model attachment and training)."""
import functools

import torchmetrics
import numpy as np
import torch
import torch.nn

import curiosidade

from . import train_test_models


def test_probe_torch_ff(fixture_pretrained_torch_ff: torch.nn.Module):
    df_train, df_eval, df_test, num_labels = train_test_models.gen_random_dataset()

    probing_model_fn = curiosidade.probers.utils.get_probing_model_feedforward(
        hidden_layer_dims=[30],
    )

    acc_fn = torchmetrics.Accuracy(num_classes=num_labels)
    f1_fn = torchmetrics.F1Score(num_classes=num_labels)

    def metrics_fn(logits: torch.Tensor, truth_labels: torch.Tensor) -> dict[str, float]:
        accuracy = acc_fn(logits, truth_labels).detach().cpu().item()
        f1 = f1_fn(logits, truth_labels).detach().cpu().item()
        return {"accuracy": accuracy, "f1": f1}

    batch_size = 16

    probing_dataloader_train = torch.utils.data.DataLoader(
        df_train, batch_size=batch_size, shuffle=True
    )
    probing_dataloader_eval = torch.utils.data.DataLoader(
        df_eval, batch_size=batch_size, shuffle=False
    )
    probing_dataloader_test = torch.utils.data.DataLoader(
        df_test, batch_size=batch_size, shuffle=False
    )

    task = curiosidade.ProbingTaskCustom(
        probing_dataloader_train=probing_dataloader_train,
        probing_dataloader_eval=probing_dataloader_eval,
        probing_dataloader_test=probing_dataloader_test,
        loss_fn=torch.nn.CrossEntropyLoss(),
        task_name="test task",
        task_type="classification",
        output_dim=num_labels,
        metrics_fn=metrics_fn,
    )

    probing_factory = curiosidade.ProbingModelFactory(
        probing_model_fn=probing_model_fn,
        optim_fn=functools.partial(torch.optim.Adam, lr=0.005),
        task=task,
    )

    prober_container = curiosidade.attach_probers(
        base_model=fixture_pretrained_torch_ff,
        probing_model_factory=probing_factory,
        modules_to_attach=["params.relu1", "params.relu3"],
        random_seed=32,
        prune_unrelated_modules="infer",
    )

    probing_results = prober_container.train(num_epochs=30, show_progress_bar=None)

    df_train, df_eval, df_test = probing_results.to_pandas(
        aggregate_by=["batch_index"],
        aggregate_fn=[np.min, np.max, np.mean],
    )

    loss_train = df_train.loc[df_train["metric_name"] == "loss", ("metric", "amin")].tolist()
    loss_eval = df_eval.loc[df_eval["metric_name"] == "loss", ("metric", "amin")].tolist()
    loss_test = df_test.loc[df_test["metric_name"] == "loss", ("metric", "amin")].tolist()

    accuracy_train = df_train.loc[
        df_train["metric_name"] == "accuracy", ("metric", "amax")
    ].tolist()
    accuracy_eval = df_eval.loc[df_eval["metric_name"] == "accuracy", ("metric", "amax")].tolist()
    accuracy_test = df_test.loc[df_test["metric_name"] == "accuracy", ("metric", "amax")].tolist()

    f1_train = df_train.loc[df_train["metric_name"] == "f1", ("metric", "amax")].tolist()
    f1_eval = df_eval.loc[df_eval["metric_name"] == "f1", ("metric", "amax")].tolist()
    f1_test = df_test.loc[df_test["metric_name"] == "f1", ("metric", "amax")].tolist()

    assert loss_train[-1] < loss_train[0] * 0.1
    assert loss_eval[-1] < loss_eval[0] * 0.1
    assert accuracy_train[-1] > accuracy_train[0] * 0.5
    assert accuracy_eval[-1] > accuracy_eval[0] * 0.5
    assert f1_train[-1] > f1_train[0] * 0.5
    assert f1_eval[-1] > f1_eval[0] * 0.5

    assert abs(min(loss_test) - loss_train[-1]) < loss_train[0] * 0.1
    assert f1_test[-1] > 0.80
    assert accuracy_test[-1] > 0.80
