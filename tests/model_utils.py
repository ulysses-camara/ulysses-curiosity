import pickle
import os

import torch.nn


PRETRAINED_MODEL_DIR = os.path.join(os.path.dirname(__file__), "pretrained_models_for_tests_dir")


def load_pickled_model(model_name: str) -> torch.nn.Module:
    model_uri = os.path.join(PRETRAINED_MODEL_DIR, model_name)

    with open(model_uri, "rb") as f_in_b:
        model = pickle.load(f_in_b)

    return model


def pickle_model(model: torch.nn.Module, model_name: str) -> None:
    os.makedirs(PRETRAINED_MODEL_DIR, exist_ok=True)
    model_uri = os.path.join(PRETRAINED_MODEL_DIR, model_name)

    with open(model_uri, "wb") as f_out_b:
        pickle.dump(model, f_out_b, protocol=pickle.HIGHEST_PROTOCOL)
