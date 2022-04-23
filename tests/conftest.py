"""Test fixtures."""
import torch
import torch.nn
import pytest
import transformers
import sentence_transformers

from . import train_test_models
from . import model_utils


@pytest.fixture(scope="session", name="fixture_pretrained_torch_ff")
def fn_fixture_pretrained_torch_ff() -> torch.nn.Module:
    pretrained_torch_ff_name = "torch_ff.pt"

    try:
        model = model_utils.load_pickled_model(pretrained_torch_ff_name)

    except (OSError, FileNotFoundError):
        model = train_test_models.train(model_name=pretrained_torch_ff_name, save=True)

    return model


@pytest.fixture(scope="session", name="fixture_pretrained_torch_bifurcation")
def fn_fixture_pretrained_torch_bifurcation() -> torch.nn.Module:
    pretrained_torch_bifurcation_name = "torch_bifurcation.pt"

    try:
        model = model_utils.load_pickled_model(pretrained_torch_bifurcation_name)

    except (OSError, FileNotFoundError):
        model = train_test_models.train(model_name=pretrained_torch_bifurcation_name, save=True)

    return model


@pytest.fixture(scope="session", name="fixture_pretrained_distilbert")
def fn_fixture_pretrained_distilbert() -> tuple[
    transformers.PreTrainedModel, transformers.DistilBertTokenizer
]:
    tokenizer = transformers.DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased", cache_dir=model_utils.PRETRAINED_MODEL_DIR
    )
    distilbert = transformers.DistilBertModel.from_pretrained(
        "distilbert-base-uncased", cache_dir=model_utils.PRETRAINED_MODEL_DIR
    )
    return distilbert, tokenizer


@pytest.fixture(scope="session", name="fixture_pretrained_minilmv2")
def fn_fixture_pretrained_minilmv2():
    minilmv2 = sentence_transformers.SentenceTransformer(
        "paraphrase-MiniLM-L6-v2", device="cpu", cache_folder=model_utils.PRETRAINED_MODEL_DIR
    )
    tokenizer = minilmv2.tokenizer
    distilbert = minilmv2.get_submodule("0")
    return distilbert, tokenizer
