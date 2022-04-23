"""Test main features from package (probing model attachment and training)."""
import functools
import warnings

import torchmetrics
import numpy as np
import torch
import torch.nn
import transformers
import datasets
import sentence_transformers

import curiosidade

from . import train_test_models
from . import architectures


def standard_result_validation(
    probing_results,
    scale_loss: float = 0.1,
    scale_accuracy: float = 0.5,
    scale_f1: float = 0.5,
    min_f1_test: float = 0.8,
    min_accuracy_test: float = 0.8,
):
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

    assert loss_train[-1] <= loss_train[0] * scale_loss
    assert loss_eval[-1] <= loss_eval[0] * scale_loss
    assert accuracy_train[-1] >= accuracy_train[0] * scale_accuracy
    assert accuracy_eval[-1] >= accuracy_eval[0] * scale_accuracy
    assert f1_train[-1] >= f1_train[0] * scale_f1
    assert f1_eval[-1] >= f1_eval[0] * scale_f1

    assert abs(min(loss_test) - loss_train[-1]) <= loss_train[0] * scale_loss
    assert f1_test[-1] >= min_f1_test
    assert accuracy_test[-1] >= min_accuracy_test


def test_probe_torch_ff(fixture_pretrained_torch_ff: torch.nn.Module):
    df_train, df_eval, df_test, num_labels = train_test_models.gen_random_dataset()

    probing_model_fn = curiosidade.probers.utils.get_probing_model_feedforward(
        hidden_layer_dims=[30],
    )

    acc_fn = torchmetrics.Accuracy(num_classes=num_labels)
    f1_fn = torchmetrics.F1Score(num_classes=num_labels)

    def metrics_fn(logits: torch.Tensor, truth_labels: torch.Tensor) -> dict[str, float]:
        # pylint: disable='not-callable'
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
        task_name="test task (torch, simple)",
        task_type="classification",
        output_dim=num_labels,
        metrics_fn=metrics_fn,
    )

    probing_factory = curiosidade.ProbingModelFactory(
        probing_model_fn=probing_model_fn,
        optim_fn=functools.partial(torch.optim.Adam, lr=0.005),
        task=task,
    )

    with warnings.catch_warnings():
        warnings.simplefilter(action="error", category=UserWarning)
        prober_container = curiosidade.attach_probers(
            base_model=fixture_pretrained_torch_ff,
            probing_model_factory=probing_factory,
            modules_to_attach=["params.relu1", "params.relu3"],
            random_seed=32,
            prune_unrelated_modules=["params.lin4"],
        )

    assert prober_container.pruned_modules == ("params.lin4",)
    assert prober_container.probed_modules == ("params.relu1", "params.relu3")

    probing_results = prober_container.train(num_epochs=30, show_progress_bar=None)

    standard_result_validation(probing_results)


def test_probe_torch_bifurcation(fixture_pretrained_torch_bifurcation: torch.nn.Module):
    df_train, df_eval, df_test, num_labels = train_test_models.gen_random_dataset()

    probing_model_fn = architectures.ProbingModelBifurcation

    acc_fn = torchmetrics.Accuracy(num_classes=num_labels)
    f1_fn = torchmetrics.F1Score(num_classes=num_labels)

    def metrics_fn(logits: torch.Tensor, truth_labels: torch.Tensor) -> dict[str, float]:
        # pylint: disable='not-callable'
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
        task_name="test task (torch. bifurcation)",
        task_type="classification",
        output_dim=num_labels,
        metrics_fn=metrics_fn,
    )

    probing_factory = curiosidade.ProbingModelFactory(
        probing_model_fn=probing_model_fn,
        optim_fn=functools.partial(torch.optim.Adam, lr=0.005),
        task=task,
    )

    with warnings.catch_warnings():
        warnings.simplefilter(action="error", category=UserWarning)
        prober_container = curiosidade.attach_probers(
            base_model=fixture_pretrained_torch_bifurcation,
            probing_model_factory=probing_factory,
            modules_to_attach="params_a.bifurcation",
            random_seed=32,
            prune_unrelated_modules="infer",
        )

    assert prober_container.pruned_modules == ("params_b",)
    assert prober_container.probed_modules == ("params_a.bifurcation",)

    probing_results = prober_container.train(num_epochs=30, show_progress_bar=None)

    standard_result_validation(probing_results)


def load_dataset_imdb(
    tokenizer,
) -> tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset, int]:
    dataset_train, dataset_test = datasets.load_dataset("imdb", split=["train", "test"])

    def tokenize_fn(item):
        sentlen_label = min(len(tokenizer.encode(item["text"])), 512)
        sentlen_label = float(max(1, np.ceil(sentlen_label / 64) - 1))

        new_item = tokenizer(item["text"], truncation=True, padding="max_length")
        new_item["label"] = sentlen_label

        return new_item

    num_classes = 8

    dataset_train = dataset_train.shard(num_shards=50, index=0)
    dataset_eval = dataset_test.shard(num_shards=50, index=0)
    dataset_test = dataset_test.shard(num_shards=50, index=2)

    dataset_train = dataset_train.map(tokenize_fn, remove_columns="text")
    dataset_eval = dataset_eval.map(tokenize_fn, remove_columns="text")
    dataset_test = dataset_test.map(tokenize_fn, remove_columns="text")

    dataset_train.set_format("torch")
    dataset_eval.set_format("torch")
    dataset_test.set_format("torch")

    return dataset_train, dataset_eval, dataset_test, num_classes


def test_probe_distilbert(
    fixture_pretrained_distilbert: tuple[
        transformers.PreTrainedModel, transformers.DistilBertTokenizer
    ],
):
    distilbert, tokenizer = fixture_pretrained_distilbert
    dataset_train, dataset_eval, dataset_test, num_classes = load_dataset_imdb(tokenizer)

    probing_dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=32,
        shuffle=True,
    )

    probing_dataloader_eval = torch.utils.data.DataLoader(
        dataset_eval,
        batch_size=32,
        shuffle=False,
    )

    probing_dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=32,
        shuffle=False,
    )

    probing_model_fn = curiosidade.probers.utils.get_probing_model_for_sequences(
        hidden_layer_dims=[192, 64],
        pooling_strategy="mean",
    )

    acc_fn = torchmetrics.Accuracy(num_classes=num_classes).to("cuda")
    f1_fn = torchmetrics.F1Score(num_classes=num_classes).to("cuda")

    def metrics_fn(logits, truth_labels):
        # pylint: disable='not-callable'
        acc = acc_fn(logits, truth_labels).detach().cpu().item()
        f1 = f1_fn(logits, truth_labels).detach().cpu().item()
        return {"accuracy": acc, "f1": f1}

    task = curiosidade.ProbingTaskCustom(
        probing_dataloader_train=probing_dataloader_train,
        probing_dataloader_eval=probing_dataloader_eval,
        probing_dataloader_test=probing_dataloader_test,
        loss_fn=torch.nn.CrossEntropyLoss(),
        task_name="test distilbert sentlen",
        output_dim=num_classes,
        metrics_fn=metrics_fn,
    )

    optim_fn = functools.partial(torch.optim.Adam, lr=0.001)
    lr_scheduler_fn = functools.partial(torch.optim.lr_scheduler.ExponentialLR, gamma=0.9)

    probing_factory = curiosidade.ProbingModelFactory(
        task=task,
        probing_model_fn=probing_model_fn,
        optim_fn=optim_fn,
        lr_scheduler_fn=lr_scheduler_fn,
    )

    with warnings.catch_warnings():
        warnings.simplefilter(action="error", category=UserWarning)
        prober_container = curiosidade.core.attach_probers(
            base_model=distilbert,
            probing_model_factory=probing_factory,
            modules_to_attach="transformer.layer.[02]",
            device="cuda" if torch.cuda.is_available() else "cpu",
            prune_unrelated_modules="infer",
        )

    assert prober_container.pruned_modules
    assert prober_container.probed_modules == ("transformer.layer.0", "transformer.layer.2")

    probing_results = prober_container.train(
        num_epochs=3,
        show_progress_bar="epoch",
        gradient_accumulation_steps=2,
    )

    standard_result_validation(probing_results, scale_loss=0.7)


def test_probe_sentence_minilmv2(
    fixture_pretrained_minilmv2: tuple[
        sentence_transformers.models.Transformer, transformers.DistilBertTokenizer
    ],
):
    minilmv2, tokenizer = fixture_pretrained_minilmv2
    dataset_train, dataset_eval, dataset_test, num_classes = load_dataset_imdb(tokenizer)

    probing_dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=32,
        shuffle=True,
    )

    probing_dataloader_eval = torch.utils.data.DataLoader(
        dataset_eval,
        batch_size=32,
        shuffle=False,
    )

    probing_dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=32,
        shuffle=False,
    )

    probing_model_fn = curiosidade.probers.utils.get_probing_model_for_sequences(
        hidden_layer_dims=[256, 128],
        pooling_strategy="mean",
    )

    acc_fn = torchmetrics.Accuracy(num_classes=num_classes).to("cuda")
    f1_fn = torchmetrics.F1Score(num_classes=num_classes).to("cuda")

    def metrics_fn(logits, truth_labels):
        # pylint: disable='not-callable'
        acc = acc_fn(logits, truth_labels).detach().cpu().item()
        f1 = f1_fn(logits, truth_labels).detach().cpu().item()
        return {"accuracy": acc, "f1": f1}

    task = curiosidade.ProbingTaskCustom(
        probing_dataloader_train=probing_dataloader_train,
        probing_dataloader_eval=probing_dataloader_eval,
        probing_dataloader_test=probing_dataloader_test,
        loss_fn=torch.nn.CrossEntropyLoss(),
        task_name="test sentence minilmv2 sentlen",
        output_dim=num_classes,
        metrics_fn=metrics_fn,
    )

    optim_fn = functools.partial(torch.optim.Adam, lr=0.05)
    lr_scheduler_fn = functools.partial(torch.optim.lr_scheduler.ExponentialLR, gamma=0.8)

    probing_factory = curiosidade.ProbingModelFactory(
        task=task,
        probing_model_fn=probing_model_fn,
        optim_fn=optim_fn,
        lr_scheduler_fn=lr_scheduler_fn,
    )

    with warnings.catch_warnings():
        warnings.simplefilter(action="error", category=UserWarning)
        prober_container = curiosidade.core.attach_probers(
            base_model=minilmv2,
            probing_model_factory=probing_factory,
            modules_to_attach="auto_model.encoder.layer.[02]",
            device="cuda" if torch.cuda.is_available() else "cpu",
            prune_unrelated_modules="infer",
        )

    assert len(prober_container.pruned_modules) > 40
    assert prober_container.probed_modules == (
        "auto_model.encoder.layer.0",
        "auto_model.encoder.layer.2",
    )

    probing_results = prober_container.train(
        num_epochs=4,
        show_progress_bar="epoch",
        gradient_accumulation_steps=2,
    )

    standard_result_validation(probing_results, scale_loss=0.6)
