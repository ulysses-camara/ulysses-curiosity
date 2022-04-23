"""Test expected warnings."""
import pytest
import torch
import torch.nn

import curiosidade

from . import train_test_models


def test_warning_attachment_fail(fixture_pretrained_torch_ff: torch.nn.Module):
    df_train, _, _, num_labels = train_test_models.gen_random_dataset(
        train_size=10,
        eval_size=1,
        test_size=1,
    )

    probing_model_fn = curiosidade.probers.utils.get_probing_model_feedforward(
        hidden_layer_dims=[10],
    )

    probing_dataloader_train = torch.utils.data.DataLoader(df_train, batch_size=4, shuffle=True)

    task = curiosidade.ProbingTaskCustom(
        probing_dataloader_train=probing_dataloader_train,
        loss_fn=torch.nn.CrossEntropyLoss(),
        task_name="test task (torch, simple)",
        task_type="classification",
        output_dim=num_labels,
    )

    probing_factory = curiosidade.ProbingModelFactory(
        probing_model_fn=probing_model_fn,
        optim_fn=torch.optim.Adam,
        task=task,
    )

    with pytest.warns(
        UserWarning, match="Some of the provided modules were not effectively attached:"
    ):
        prober_container = curiosidade.attach_probers(
            base_model=fixture_pretrained_torch_ff,
            probing_model_factory=probing_factory,
            modules_to_attach=["params.relu", "params.relu3", "incorrect_label"],
            random_seed=32,
            prune_unrelated_modules=None,
        )

    assert prober_container.probed_modules == ("params.relu3",)


def test_warning_prune_fail(fixture_pretrained_torch_ff: torch.nn.Module):
    df_train, _, _, num_labels = train_test_models.gen_random_dataset(
        train_size=10,
        eval_size=1,
        test_size=1,
    )

    probing_model_fn = curiosidade.probers.utils.get_probing_model_feedforward(
        hidden_layer_dims=[10],
    )

    probing_dataloader_train = torch.utils.data.DataLoader(df_train, batch_size=4, shuffle=True)

    task = curiosidade.ProbingTaskCustom(
        probing_dataloader_train=probing_dataloader_train,
        loss_fn=torch.nn.CrossEntropyLoss(),
        task_name="test task (torch, simple)",
        task_type="classification",
        output_dim=num_labels,
    )

    probing_factory = curiosidade.ProbingModelFactory(
        probing_model_fn=probing_model_fn,
        optim_fn=torch.optim.Adam,
        task=task,
    )

    with pytest.warns(
        UserWarning, match="Some of modules to prune were not found in pretrained model:"
    ):
        prober_container = curiosidade.attach_probers(
            base_model=fixture_pretrained_torch_ff,
            probing_model_factory=probing_factory,
            modules_to_attach=r"params.relu\d+",
            random_seed=32,
            prune_unrelated_modules=["invalid_module_to_prune"],
        )

    assert prober_container.probed_modules
    assert prober_container.pruned_modules == tuple()


def test_warning_retraining_probing_models(fixture_pretrained_torch_ff: torch.nn.Module):
    df_train, _, _, num_labels = train_test_models.gen_random_dataset(
        train_size=10,
        eval_size=1,
        test_size=1,
    )

    probing_model_fn = curiosidade.probers.utils.get_probing_model_feedforward(
        hidden_layer_dims=[10],
    )

    probing_dataloader_train = torch.utils.data.DataLoader(df_train, batch_size=4, shuffle=True)

    task = curiosidade.ProbingTaskCustom(
        probing_dataloader_train=probing_dataloader_train,
        loss_fn=torch.nn.CrossEntropyLoss(),
        task_name="test task (torch, simple)",
        task_type="classification",
        output_dim=num_labels,
    )

    probing_factory = curiosidade.ProbingModelFactory(
        probing_model_fn=probing_model_fn,
        optim_fn=torch.optim.Adam,
        task=task,
    )

    prober_container = curiosidade.attach_probers(
        base_model=fixture_pretrained_torch_ff,
        probing_model_factory=probing_factory,
        modules_to_attach=["params.relu", "params.relu3"],
        random_seed=32,
        prune_unrelated_modules="infer",
    )

    prober_container.train(num_epochs=1)

    with pytest.warns(
        UserWarning, match="Probing weights are already pretrained from previous run."
    ):
        prober_container.train(num_epochs=1)


def test_error_train_without_attachment(fixture_pretrained_torch_ff: torch.nn.Module):
    df_train, _, _, num_labels = train_test_models.gen_random_dataset(
        train_size=10,
        eval_size=1,
        test_size=1,
    )

    probing_model_fn = curiosidade.probers.utils.get_probing_model_feedforward(
        hidden_layer_dims=[10],
    )

    probing_dataloader_train = torch.utils.data.DataLoader(df_train, batch_size=4, shuffle=True)

    task = curiosidade.ProbingTaskCustom(
        probing_dataloader_train=probing_dataloader_train,
        loss_fn=torch.nn.CrossEntropyLoss(),
        task_name="test task (torch, simple)",
        task_type="classification",
        output_dim=num_labels,
    )

    probing_factory = curiosidade.ProbingModelFactory(
        probing_model_fn=probing_model_fn,
        optim_fn=torch.optim.Adam,
        task=task,
    )

    prober_container = curiosidade.attach_probers(
        base_model=fixture_pretrained_torch_ff,
        probing_model_factory=probing_factory,
        modules_to_attach="invalid-pattern",
        random_seed=32,
        prune_unrelated_modules="infer",
    )

    with pytest.raises(RuntimeError):
        prober_container.train(num_epochs=1)
