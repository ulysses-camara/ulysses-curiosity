"""Test main features from package (probing model attachment and training)."""
import functools
import warnings

import pytest
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


def standard_result_validation(probing_results):
    def fn_test(x):
        return 1.0 / (1.0 + float(np.sum(x)))

    df_train, df_eval, df_test = probing_results.to_pandas(
        aggregate_by=["batch_index"],
        aggregate_fn=[np.min, np.mean, np.max, lambda _: 5, fn_test],
    )

    kwargs = dict(n=1, axis=-1)
    cols = [("metric", "amin"), ("metric", "amin"), ("metric", "amax")]

    assert np.all(np.diff(df_train[cols], **kwargs) >= 0.0)
    assert np.all(np.diff(df_eval[cols], **kwargs) >= 0.0)
    assert np.all(np.diff(df_test[cols], **kwargs) >= 0.0)

    assert np.allclose(df_train[("metric", "<lambda_0>")], 5)
    assert np.allclose(df_eval[("metric", "<lambda_0>")], 5)
    assert np.allclose(df_test[("metric", "<lambda_0>")], 5)

    assert np.all(df_train[("metric", "fn_test")] <= 1.0)
    assert np.all(df_eval[("metric", "fn_test")] <= 1.0)
    assert np.all(df_test[("metric", "fn_test")] <= 1.0)


def test_probe_torch_lstm_onedir_1_layer(
    fixture_pretrained_torch_lstm_onedir_1_layer: torch.nn.Module,
):
    df_train, df_eval, df_test, num_classes = train_test_models.gen_random_dataset(
        integer_data=True,
        vocab_size=fixture_pretrained_torch_lstm_onedir_1_layer.vocab_size,
    )

    probing_model_fn = curiosidade.probers.utils.get_probing_model_for_sequences(
        hidden_layer_dims=[128],
    )

    if num_classes >= 3:
        acc_fn = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes)
        f1_fn = torchmetrics.classification.MulticlassF1Score(num_classes=num_classes)

    else:
        acc_fn = torchmetrics.BinaryAccuracy()
        f1_fn = torchmetrics.BinaryF1Score()

    acc_fn = acc_fn.to("cpu")
    f1_fn = f1_fn.to("cpu")

    def metrics_fn(logits: torch.Tensor, truth_labels: torch.Tensor) -> dict[str, float]:
        # pylint: disable='not-callable'
        accuracy = float(acc_fn(logits, truth_labels).detach().cpu().item())
        f1 = float(f1_fn(logits, truth_labels).detach().cpu().item())
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
        task_name="test task (torch, lstm, onedir, 1 layer)",
        task_type="classification",
        output_dim=num_classes,
        metrics_fn=metrics_fn,
    )

    probing_factory = curiosidade.ProbingModelFactory(
        probing_model_fn=probing_model_fn,
        optim_fn=functools.partial(torch.optim.Adam, lr=0.001),
        task=task,
    )

    with warnings.catch_warnings():
        warnings.simplefilter(action="error", category=UserWarning)
        prober_container = curiosidade.attach_probers(
            base_model=fixture_pretrained_torch_lstm_onedir_1_layer,
            probing_model_factory=probing_factory,
            modules_to_attach=["lstm"],
            random_seed=32,
            prune_unrelated_modules=["lin_out"],
        )

    assert prober_container.pruned_modules == ("lin_out",)
    assert prober_container.probed_modules == ("lstm",)

    probing_results = prober_container.train(num_epochs=40, show_progress_bar=None)

    standard_result_validation(probing_results)


def test_probe_torch_ff(fixture_pretrained_torch_ff: torch.nn.Module):
    df_train, df_eval, df_test, num_classes = train_test_models.gen_random_dataset()

    probing_model_fn = curiosidade.probers.utils.get_probing_model_feedforward(
        hidden_layer_dims=[30],
    )

    if num_classes >= 3:
        acc_fn = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes)
        f1_fn = torchmetrics.classification.MulticlassF1Score(num_classes=num_classes)

    else:
        acc_fn = torchmetrics.BinaryAccuracy()
        f1_fn = torchmetrics.BinaryF1Score()

    acc_fn = acc_fn.to("cpu")
    f1_fn = f1_fn.to("cpu")

    def metrics_fn(logits: torch.Tensor, truth_labels: torch.Tensor) -> dict[str, float]:
        # pylint: disable='not-callable'
        accuracy = float(acc_fn(logits, truth_labels).detach().cpu().item())
        f1 = float(f1_fn(logits, truth_labels).detach().cpu().item())
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
        output_dim=num_classes,
        metrics_fn=metrics_fn,
    )

    probing_factory = curiosidade.ProbingModelFactory(
        probing_model_fn=probing_model_fn,
        optim_fn=functools.partial(torch.optim.Adam, lr=0.001),
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

    probing_results = prober_container.train(num_epochs=40, show_progress_bar=None)

    standard_result_validation(probing_results)


def test_probe_torch_bifurcation(fixture_pretrained_torch_bifurcation: torch.nn.Module):
    df_train, df_eval, df_test, num_classes = train_test_models.gen_random_dataset()

    probing_model_fn = architectures.ProbingModelBifurcation

    if num_classes >= 3:
        acc_fn = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes)
        f1_fn = torchmetrics.classification.MulticlassF1Score(num_classes=num_classes)

    else:
        acc_fn = torchmetrics.BinaryAccuracy()
        f1_fn = torchmetrics.BinaryF1Score()

    acc_fn = acc_fn.to("cpu")
    f1_fn = f1_fn.to("cpu")

    def metrics_fn(logits: torch.Tensor, truth_labels: torch.Tensor) -> dict[str, float]:
        # pylint: disable='not-callable'
        accuracy = float(acc_fn(logits, truth_labels).detach().cpu().item())
        f1 = float(f1_fn(logits, truth_labels).detach().cpu().item())
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
        output_dim=num_classes,
        metrics_fn=metrics_fn,
    )

    probing_factory = curiosidade.ProbingModelFactory(
        probing_model_fn=probing_model_fn,
        optim_fn=functools.partial(torch.optim.Adam, lr=0.001),
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

    probing_results = prober_container.train(num_epochs=40, show_progress_bar=None)

    standard_result_validation(probing_results)


def load_dataset_imdb(
    tokenizer,
) -> tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset, int]:
    dataset_train, dataset_test = datasets.load_dataset("imdb", split=["train", "test"])

    def tokenize_fn(item):
        sentlen_label = min(len(tokenizer.encode(item["text"])), 512)
        sentlen_label = float(max(1, np.ceil(sentlen_label / 128) - 1))

        new_item = tokenizer(item["text"], truncation=True, padding="max_length")
        new_item["label"] = sentlen_label

        return new_item

    num_classes = 4

    dataset_train = dataset_train.shard(num_shards=200, index=0)
    dataset_eval = dataset_test.shard(num_shards=200, index=0)
    dataset_test = dataset_test.shard(num_shards=200, index=2)

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

    device = "cpu"

    if num_classes >= 3:
        acc_fn = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes)
        f1_fn = torchmetrics.classification.MulticlassF1Score(num_classes=num_classes)

    else:
        acc_fn = torchmetrics.BinaryAccuracy()
        f1_fn = torchmetrics.BinaryF1Score()

    acc_fn = acc_fn.to("cpu")
    f1_fn = f1_fn.to("cpu")

    def metrics_fn(logits, truth_labels):
        # pylint: disable='not-callable'
        acc = float(acc_fn(logits, truth_labels).detach().cpu().item())
        f1 = float(f1_fn(logits, truth_labels).detach().cpu().item())
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
            modules_to_attach="embeddings.LayerNorm|transformer.layer.0.output_layer_norm",
            device=device,
            prune_unrelated_modules="infer",
        )

    assert prober_container.pruned_modules
    assert prober_container.probed_modules == (
        "embeddings.LayerNorm",
        "transformer.layer.0.output_layer_norm",
    )

    probing_results = prober_container.train(
        num_epochs=2,
        show_progress_bar="epoch",
        gradient_accumulation_steps=3,
    )

    standard_result_validation(probing_results)


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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if num_classes >= 3:
        acc_fn = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes)
        f1_fn = torchmetrics.classification.MulticlassF1Score(num_classes=num_classes)

    else:
        acc_fn = torchmetrics.BinaryAccuracy()
        f1_fn = torchmetrics.BinaryF1Score()

    acc_fn = acc_fn.to("cpu")
    f1_fn = f1_fn.to("cpu")

    def metrics_fn(logits, truth_labels):
        # pylint: disable='not-callable'
        acc = float(acc_fn(logits, truth_labels).detach().cpu().item())
        f1 = float(f1_fn(logits, truth_labels).detach().cpu().item())
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
            base_model=minilmv2,
            probing_model_factory=probing_factory,
            modules_to_attach="auto_model.encoder.layer.[02]",
            device=device,
            prune_unrelated_modules="infer",
        )

    assert len(prober_container.pruned_modules) > 40
    assert prober_container.probed_modules == (
        "auto_model.encoder.layer.0",
        "auto_model.encoder.layer.2",
    )

    probing_results = prober_container.train(
        num_epochs=2,
        show_progress_bar="epoch",
        gradient_accumulation_steps=2,
    )

    standard_result_validation(probing_results)


@pytest.mark.parametrize(
    "pooling_strategy,embedding_index_to_keep,expected_tensor",
    (
        ("mean", 0, [[6.0, 7.0, 8.0], [21.0, 22.0, 23.0]]),
        ("mean", -1, [[6.0, 7.0, 8.0], [21.0, 22.0, 23.0]]),
        ("max", 0, [[12.0, 13.0, 14.0], [27.0, 28.0, 29.0]]),
        ("max", 3, [[12.0, 13.0, 14.0], [27.0, 28.0, 29.0]]),
        ("keep_single_index", 0, [[0.0, 1.0, 2.0], [15.0, 16.0, 17.0]]),
        ("keep_single_index", 2, [[6.0, 7.0, 8.0], [21.0, 22.0, 23.0]]),
    ),
)
def test_standard_prober_for_sequences_pooling_strategies(
    pooling_strategy: str, embedding_index_to_keep: int, expected_tensor: list[list[float]]
):
    fn_prober = curiosidade.probers.utils.get_probing_model_for_sequences(
        hidden_layer_dims=[6],
        pooling_strategy=pooling_strategy,
        pooling_axis=1,
        embedding_index_to_keep=embedding_index_to_keep,
    )

    prober = fn_prober(input_dim=3, output_dim=1)
    assert prober.embedding_index_to_keep == embedding_index_to_keep

    pooling_fn = prober.pooling_fn
    input_tensor = torch.arange(0, 2 * 5 * 3).view(2, 5, 3).float()

    output_tensor = pooling_fn(input_tensor)
    assert output_tensor.allclose(torch.tensor(expected_tensor, dtype=torch.float))


def test_invalid_pooling_strategy():
    with pytest.raises(ValueError):
        fn_prober = curiosidade.probers.utils.get_probing_model_for_sequences(
            hidden_layer_dims=[10],
            pooling_strategy="invalid",
        )

        prober = fn_prober(1, 1)
