import pytest
import transformers
import sentence_transformers

from . import train_test_models
from . import model_utils


@pytest.fixture(scope="session", name="fixture_pretrained_torch_ff")
def fn_fixture_pretrained_torch_ff():
    pretrained_torch_ff_name = "torch_ff.pt"

    try:
        model = model_utils.load_pickled_model(pretrained_torch_ff_name)

    except (OSError, FileNotFoundError):
        model = train_test_models.train(model_name=pretrained_torch_ff_name, save=True)

    return model


@pytest.fixture(scope="session", name="fixture_pretrained_torch_bifurcation")
def fn_fixture_pretrained_torch_bifurcation():
    pretrained_torch_bifurcation_name = "torch_bifurcation.pt"

    try:
        model = load_pickled_model(pretrained_torch_bifurcation_name)

    except (OSError, FileNotFoundError):
        model = train_test_models.train(model_name=pretrained_torch_bifurcation_name, save=True)

    return model


@pytest.fixture(scope="session", name="fixture_pretrained_distilbert")
def fn_fixture_pretrained_distilbert():
    pretrained_distilbert_name = ""
    return model


@pytest.fixture(scope="session", name="fixture_pretrained_sbert")
def fn_fixture_pretrained_sbert():
    pretrained_sbert_name = ""
    return model
