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
from . import model_utils


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

    prober_container.detach()
    prober_container.detach()  # Double detach to make sure no warning will be raised.


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

    prober_container.detach()
    prober_container.detach()  # Double detach to make sure no warning will be raised.


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

    prober_container.detach()
    prober_container.detach()  # Double detach to make sure no warning will be raised.


def load_dataset_imdb(
    tokenizer,
) -> tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset, int]:
    dataset_train, dataset_test = datasets.load_dataset(
        "imdb", split=["train", "test"], cache_dir="cache"
    )

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

    prober_container.detach()
    prober_container.detach()  # Double detach to make sure no warning will be raised.


def test_probe_sentence_minilmv2(
    fixture_pretrained_minilmv2: sentence_transformers.SentenceTransformer,
):
    minilmv2 = fixture_pretrained_minilmv2.get_submodule("0")
    tokenizer = fixture_pretrained_minilmv2.tokenizer

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

    prober_container.detach()
    prober_container.detach()  # Double detach to make sure no warning will be raised.


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

    device = "cpu"

    prober = fn_prober(input_dim=3, output_dim=1).to(device)
    assert prober.embedding_index_to_keep == embedding_index_to_keep

    pooling_fn = prober.pooling_fn
    input_tensor = torch.arange(0, 2 * 5 * 3).view(2, 5, 3).float().to(device)

    output_tensor = pooling_fn(input_tensor).detach().to("cpu")
    assert output_tensor.allclose(torch.tensor(expected_tensor, dtype=torch.float))


def test_invalid_pooling_strategy():
    with pytest.raises(ValueError):
        fn_prober = curiosidade.probers.utils.get_probing_model_for_sequences(
            hidden_layer_dims=[10],
            pooling_strategy="invalid",
        )

        fn_prober(1, 1)


def test_probe_sentence_minilmv2_full_sbert(
    fixture_pretrained_minilmv2: sentence_transformers.SentenceTransformer,
):
    dataset_train, dataset_test = datasets.load_dataset(
        "imdb", split=["train", "test"], cache_dir="cache"
    )
    buckets = np.asfarray([582.0, 987.5, 2685.3])

    def preprocess_fn(item):
        sentlen_label = len(item["text"])
        sentlen_label = float(np.digitize(len(item["text"]), buckets))

        new_item = {
            "sentence": item["text"],
            "label": sentlen_label,
        }

        return new_item

    num_classes = 4

    dataset_train = dataset_train.shard(num_shards=1000, index=0)
    dataset_eval = dataset_test.shard(num_shards=1000, index=0)
    dataset_test = dataset_test.shard(num_shards=1000, index=2)

    dataset_train = dataset_train.map(preprocess_fn)
    dataset_eval = dataset_eval.map(preprocess_fn)
    dataset_test = dataset_test.map(preprocess_fn)

    assert len(dataset_train)
    assert len(dataset_eval)
    assert len(dataset_test)

    cols = ["sentence", "label"]
    dataset_train = dataset_train.to_pandas()[cols].values.tolist()
    dataset_eval = dataset_eval.to_pandas()[cols].values.tolist()
    dataset_test = dataset_test.to_pandas()[cols].values.tolist()

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

    probing_model_fn = functools.partial(
        model_utils.SwitchableProber,
        hidden_layer_dims=[128],
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

    modules_to_attach = [
        "0",
        "1",
        "0.auto_model.encoder.layer.5.output.LayerNorm",
        "0.auto_model.encoder",
        "0.auto_model.embeddings",
    ]

    with warnings.catch_warnings():
        warnings.simplefilter(action="error", category=UserWarning)
        prober_container = curiosidade.core.attach_probers(
            base_model=fixture_pretrained_minilmv2,
            probing_model_factory=probing_factory,
            modules_to_attach=modules_to_attach,
            device=device,
            prune_unrelated_modules="infer",
            modules_input_dim={
                "0": 384,
                "1": 384,
            },
        )

    assert len(prober_container.pruned_modules) == 0
    assert sorted(prober_container.probed_modules) == sorted(modules_to_attach)

    probing_results = prober_container.train(
        num_epochs=1,
        show_progress_bar="epoch",
    )

    standard_result_validation(probing_results)

    prober_container.detach()
    prober_container.detach()  # Double detach to make sure no warning will be raised.


@pytest.mark.parametrize(
    "num_epochs,batch_size,gradient_accumulation_steps,num_probers,"
    "include_eval,include_test,include_lr_scheduler",
    (
        (1, 10, 1, 1, False, False, False),
        (1, 20, 1, 1, False, False, False),
        (2, 20, 1, 1, False, False, False),
        (1, 10, 2, 1, False, False, False),
        (1, 20, 2, 1, False, False, False),
        (2, 20, 2, 1, False, False, False),
        (1, 10, 1, 2, False, False, False),
        (1, 20, 1, 2, False, False, False),
        (2, 20, 1, 2, False, False, False),
        (1, 10, 2, 2, False, False, False),
        (1, 20, 2, 2, False, False, False),
        (2, 20, 2, 2, False, False, False),
        (1, 10, 1, 1, True, False, False),
        (1, 20, 1, 1, True, False, False),
        (2, 20, 1, 1, True, False, False),
        (1, 10, 2, 1, True, False, False),
        (1, 20, 2, 1, True, False, False),
        (2, 20, 2, 1, True, False, False),
        (1, 10, 1, 2, True, False, False),
        (1, 20, 1, 2, True, False, False),
        (2, 20, 1, 2, True, False, False),
        (1, 10, 1, 1, True, True, False),
        (1, 20, 1, 1, True, True, False),
        (2, 20, 1, 1, True, True, False),
        (1, 10, 2, 1, True, True, False),
        (1, 20, 2, 1, True, True, False),
        (2, 20, 2, 1, True, True, False),
        (1, 10, 1, 2, True, True, False),
        (1, 20, 1, 2, True, True, False),
        (2, 20, 1, 2, True, True, False),
        (1, 10, 1, 1, True, True, True),
        (1, 20, 1, 1, True, True, True),
        (2, 20, 1, 1, True, True, True),
        (1, 10, 2, 1, True, True, True),
        (1, 20, 2, 1, True, True, True),
        (2, 20, 2, 1, True, True, True),
        (1, 10, 1, 2, True, True, True),
        (1, 20, 1, 2, True, True, True),
        (2, 20, 1, 2, True, True, True),
        (1, 10, 2, 2, True, True, True),
        (1, 20, 2, 2, True, True, True),
        (2, 20, 2, 2, True, True, True),
        (1, 10, 1, 1, False, False, True),
        (1, 20, 1, 1, False, False, True),
        (2, 20, 1, 1, False, False, True),
        (1, 10, 2, 1, False, False, True),
        (1, 20, 2, 1, False, False, True),
        (2, 20, 2, 1, False, False, True),
        (1, 10, 1, 2, False, False, True),
        (1, 20, 1, 2, False, False, True),
        (2, 20, 1, 2, False, False, True),
        (1, 10, 2, 2, False, False, True),
        (1, 20, 2, 2, False, False, True),
        (2, 20, 2, 2, False, False, True),
        (2, 30, 2, 2, False, False, True),
        (6, 30, 2, 2, False, False, True),
        (6, 30, 2, 3, False, False, True),
        (6, 30, 1, 3, False, False, True),
        (5, 30, 1, 3, False, True, True),
        (6, 30, 2, 2, False, False, False),
        (6, 30, 2, 3, False, False, False),
        (6, 30, 1, 3, False, False, False),
        (5, 30, 1, 3, False, True, False),
        (3, 10, 1, 3, True, False, False),
        (2, 3, 2, 2, False, False, True),
        (6, 3, 2, 2, False, False, True),
        (6, 3, 2, 3, False, False, True),
        (6, 3, 1, 3, False, False, True),
        (5, 3, 1, 3, False, True, True),
        (6, 3, 2, 2, False, False, False),
        (6, 3, 2, 3, False, False, False),
        (6, 3, 1, 3, False, False, False),
        (5, 3, 1, 3, False, True, False),
        (1, 3, 1, 3, True, False, True),
        (1, 3, 1, 3, True, True, True),
        (1, 3, 2, 2, True, False, False),
    ),
)
def test_optimization_and_validation_frequencies(
    fixture_pretrained_torch_ff: torch.nn.Module,
    num_epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_probers: int,
    include_eval: bool,
    include_test: bool,
    include_lr_scheduler: bool,
):
    df_train, df_eval, df_test, num_classes = train_test_models.gen_random_dataset(
        train_size=20,
        eval_size=10,
        test_size=10,
    )

    num_optim_steps = 0
    num_scheduler_steps = 0
    num_metric_computations = 0

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
        nonlocal num_metric_computations
        num_metric_computations += 1
        accuracy = float(acc_fn(logits, truth_labels).detach().cpu().item())
        f1 = float(f1_fn(logits, truth_labels).detach().cpu().item())
        return {"accuracy": accuracy, "f1": f1}

    probing_dataloader_train = torch.utils.data.DataLoader(
        df_train,
        batch_size=batch_size,
        shuffle=True,
    )

    probing_dataloader_eval = None
    probing_dataloader_test = None

    if include_eval:
        probing_dataloader_eval = torch.utils.data.DataLoader(
            df_eval, batch_size=batch_size, shuffle=False
        )

    if include_test:
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

    lr_scheduler_fn = None

    if include_lr_scheduler:

        class CustomScheduler(torch.optim.lr_scheduler.ExponentialLR):
            # pylint: disable='missing-class-docstring'
            def step(self, *args, **kwargs):
                nonlocal num_scheduler_steps
                num_scheduler_steps += 1
                return super().step(*args, **kwargs)

        lr_scheduler_fn = functools.partial(CustomScheduler, gamma=0.9)

    class CustomOptim(torch.optim.Adam):
        # pylint: disable='missing-class-docstring'
        def step(self, *args, **kwargs):
            nonlocal num_optim_steps
            num_optim_steps += 1
            return super().step(*args, **kwargs)

    probing_factory = curiosidade.ProbingModelFactory(
        probing_model_fn=probing_model_fn,
        optim_fn=functools.partial(CustomOptim, lr=0.001),
        lr_scheduler_fn=lr_scheduler_fn,
        task=task,
    )

    with warnings.catch_warnings():
        warnings.simplefilter(action="error", category=UserWarning)
        prober_container = curiosidade.attach_probers(
            base_model=fixture_pretrained_torch_ff,
            probing_model_factory=probing_factory,
            modules_to_attach=[f"params.relu{i}" for i in range(1, 1 + num_probers)],
            random_seed=32,
            prune_unrelated_modules="infer",
        )

    assert len(prober_container.probed_modules) == num_probers

    prober_container.train(
        num_epochs=num_epochs,
        show_progress_bar=None,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    assert (
        num_optim_steps
        == int(np.ceil(len(df_train) / batch_size / gradient_accumulation_steps))
        * num_epochs
        * num_probers
    )
    assert num_scheduler_steps == int(include_lr_scheduler) * (1 + num_epochs) * num_probers
    assert num_metric_computations == (
        int(np.ceil(len(df_train) / batch_size)) * num_epochs * num_probers
        + int(include_eval) * num_epochs * num_probers
        + int(include_test) * num_probers
    )

    prober_container.detach()
    prober_container.detach()  # Double detach to make sure no warning will be raised.
