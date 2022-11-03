"""Test preconfigured probing tasks."""
import typing as t
import functools
import os

import pytest
import torch
import torch.nn
import torchmetrics
import transformers
import numpy as np

import curiosidade


@pytest.mark.parametrize(
    "fn_custom_probing_task,num_classes,data_domain",
    (
        (curiosidade.ProbingTaskSentenceLength, 6, "wikipedia-ptbr"),
        (curiosidade.ProbingTaskWordContent, 1000, "wikipedia-ptbr"),
        (curiosidade.ProbingTaskTreeDepth, 6, "wikipedia-ptbr"),
        (curiosidade.ProbingTaskTopConstituent, 20, "wikipedia-ptbr"),
        (curiosidade.ProbingTaskBigramShift, 2, "wikipedia-ptbr"),
        (curiosidade.ProbingTaskPastPresent, 2, "wikipedia-ptbr"),
        (curiosidade.ProbingTaskSubjectNumber, 2, "wikipedia-ptbr"),
        (curiosidade.ProbingTaskObjectNumber, 2, "wikipedia-ptbr"),
        (curiosidade.ProbingTaskSOMO, 2, "wikipedia-ptbr"),
        (curiosidade.ProbingTaskCoordinationInversion, 2, "wikipedia-ptbr"),
        (curiosidade.ProbingTaskSentenceLength, 5, "sp-court-cases"),
        (curiosidade.ProbingTaskWordContent, 298, "sp-court-cases"),
        (curiosidade.ProbingTaskTreeDepth, 4, "sp-court-cases"),
        (curiosidade.ProbingTaskTopConstituent, 10, "sp-court-cases"),
        (curiosidade.ProbingTaskBigramShift, 2, "sp-court-cases"),
        (curiosidade.ProbingTaskPastPresent, 2, "sp-court-cases"),
        (curiosidade.ProbingTaskSubjectNumber, 2, "sp-court-cases"),
        (curiosidade.ProbingTaskObjectNumber, 2, "sp-court-cases"),
        (curiosidade.ProbingTaskSOMO, 2, "sp-court-cases"),
        (curiosidade.ProbingTaskCoordinationInversion, 2, "sp-court-cases"),
        (curiosidade.ProbingTaskSentenceLength, 5, "leg-docs-ptbr"),
        (curiosidade.ProbingTaskWordContent, 300, "leg-docs-ptbr"),
        (curiosidade.ProbingTaskTreeDepth, 4, "leg-docs-ptbr"),
        (curiosidade.ProbingTaskTopConstituent, 10, "leg-docs-ptbr"),
        (curiosidade.ProbingTaskBigramShift, 2, "leg-docs-ptbr"),
        (curiosidade.ProbingTaskPastPresent, 2, "leg-docs-ptbr"),
        (curiosidade.ProbingTaskSubjectNumber, 2, "leg-docs-ptbr"),
        (curiosidade.ProbingTaskObjectNumber, 2, "leg-docs-ptbr"),
        (curiosidade.ProbingTaskSOMO, 2, "leg-docs-ptbr"),
        (curiosidade.ProbingTaskCoordinationInversion, 2, "leg-docs-ptbr"),
        (curiosidade.ProbingTaskSentenceLength, 5, "leg-pop-comments-ptbr"),
        (curiosidade.ProbingTaskWordContent, 300, "leg-pop-comments-ptbr"),
        (curiosidade.ProbingTaskTreeDepth, 4, "leg-pop-comments-ptbr"),
        (curiosidade.ProbingTaskTopConstituent, 10, "leg-pop-comments-ptbr"),
        (curiosidade.ProbingTaskBigramShift, 2, "leg-pop-comments-ptbr"),
        (curiosidade.ProbingTaskPastPresent, 2, "leg-pop-comments-ptbr"),
        (curiosidade.ProbingTaskSubjectNumber, 2, "leg-pop-comments-ptbr"),
        (curiosidade.ProbingTaskObjectNumber, 2, "leg-pop-comments-ptbr"),
        (curiosidade.ProbingTaskSOMO, 2, "leg-pop-comments-ptbr"),
        (curiosidade.ProbingTaskCoordinationInversion, 2, "leg-pop-comments-ptbr"),
        (curiosidade.ProbingTaskSentenceLength, 5, "political-speeches-ptbr"),
        (curiosidade.ProbingTaskWordContent, 300, "political-speeches-ptbr"),
        (curiosidade.ProbingTaskTreeDepth, 4, "political-speeches-ptbr"),
        (curiosidade.ProbingTaskTopConstituent, 10, "political-speeches-ptbr"),
        (curiosidade.ProbingTaskBigramShift, 2, "political-speeches-ptbr"),
        (curiosidade.ProbingTaskPastPresent, 2, "political-speeches-ptbr"),
        (curiosidade.ProbingTaskSubjectNumber, 2, "political-speeches-ptbr"),
        (curiosidade.ProbingTaskObjectNumber, 2, "political-speeches-ptbr"),
        (curiosidade.ProbingTaskSOMO, 2, "political-speeches-ptbr"),
        (curiosidade.ProbingTaskCoordinationInversion, 2, "political-speeches-ptbr"),
    ),
)
def test_preconfigured_probing_task(
    fixture_pretrained_torch_lstm_onedir_1_layer: torch.nn.Module,
    fixture_pretrained_bertimbau_tokenizer: transformers.AutoTokenizer,
    fn_custom_probing_task: curiosidade.probers.tasks.base.BaseProbingTask,
    num_classes: int,
    data_domain: str,
):
    torch.manual_seed(127)
    np.random.seed(89)

    ProbingModel = curiosidade.probers.utils.get_probing_model_for_sequences(
        hidden_layer_dims=[128],
        include_batch_norm=True,
    )

    def fn_text_to_tensor_for_pytorch(
        content: list[str],
        labels: list[int],
        split: t.Literal["train", "eval", "test"],
    ) -> dict[str, torch.Tensor]:
        n = 20 if split == "train" else 5

        content = content[:n]
        labels = labels[:n]

        X = fixture_pretrained_bertimbau_tokenizer(
            content,
            max_length=32,
            truncation=True,
            padding="max_length",
        )["input_ids"]

        X = torch.nn.utils.rnn.pad_sequence(
            [torch.Tensor(inst) for inst in X],
            batch_first=True,
            padding_value=0,
        )

        X %= fixture_pretrained_torch_lstm_onedir_1_layer.vocab_size
        X = X.long()

        y = torch.Tensor(labels)
        y = y.float() if num_classes == 2 else y.long()

        return torch.utils.data.TensorDataset(X, y)

    if num_classes >= 3:
        acc_fn = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes)
        f1_fn = torchmetrics.classification.MulticlassF1Score(num_classes=num_classes)

    else:
        acc_fn = torchmetrics.classification.BinaryAccuracy()
        f1_fn = torchmetrics.classification.BinaryF1Score()

    acc_fn = acc_fn.to("cpu")
    f1_fn = f1_fn.to("cpu")

    def metrics_fn(logits, truth_labels):
        # pylint: disable='not-callable'
        truth_labels = truth_labels.long()
        acc = acc_fn(logits, truth_labels).detach().cpu().item()
        f1 = f1_fn(logits, truth_labels).detach().cpu().item()
        return {"accuracy": acc, "f1": f1}

    task = fn_custom_probing_task(
        fn_raw_data_to_tensor=fn_text_to_tensor_for_pytorch,
        metrics_fn=metrics_fn,
        data_domain=data_domain,
        output_dir=os.path.join(os.path.dirname(__file__), "test_probing_datasets"),
        check_cached=False,
        timeout_limit_seconds=60,
    )

    probing_factory = curiosidade.ProbingModelFactory(
        probing_model_fn=ProbingModel,
        optim_fn=functools.partial(torch.optim.Adam, lr=0.01),
        task=task,
    )

    prober_container = curiosidade.attach_probers(
        base_model=fixture_pretrained_torch_lstm_onedir_1_layer,
        probing_model_factory=probing_factory,
        modules_to_attach="lstm",
        random_seed=16,
        device="cpu",
    )

    probing_results = prober_container.train(num_epochs=60)

    df_train, _, _ = probing_results.to_pandas(
        aggregate_by=["batch_index"],
        aggregate_fn=[np.min, np.max, np.mean],
    )

    loss_train = df_train.loc[df_train["metric_name"] == "loss", ("metric", "amin")].tolist()
    accuracy_train = df_train.loc[
        df_train["metric_name"] == "accuracy", ("metric", "amax")
    ].tolist()
    f1_train = df_train.loc[df_train["metric_name"] == "f1", ("metric", "amax")].tolist()

    assert loss_train[-1] <= loss_train[0] * 0.90
    assert accuracy_train[-1] >= accuracy_train[0] * 1.10
    assert f1_train[-1] >= f1_train[0] * 1.10
