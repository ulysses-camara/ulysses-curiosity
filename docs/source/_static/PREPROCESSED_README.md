[![Tests](https://github.com/ulysses-camara/ulysses-curiosity/actions/workflows/tests.yml/badge.svg)](https://github.com/ulysses-camara/ulysses-curiosity/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/ulysses-curiosity/badge/?version=latest)](https://ulysses-curiosity.readthedocs.io/en/latest/?badge=latest)

# Ulysses Curiosity

Ulysses Curiosity is a framework for Probing Tasks.

The term "probing task" was coined in [(Conneau et. al, 2018)](#references), referring to the act of validating a pretrained model without having validation data that resembles the distribution of interest (e.g. models trained on surrogate self-supervised tasks).

The strategy focus on the elaboration of a surrogate classification task &mdash; a "Probing Task" &mdash; which should be simple, easily interpretable, and is somehow connected to the domain of interest. Probing models are attached to the pretrained model, and optimized to solve the probing task by using as input only activations from the pretrained model. The pretrained model is kept frozen during the entire process (i.e. it is not optimized).

```{mermaid}
  :align: center
flowchart TB
P1(["Probing Model A"])
P2(["Probing Model B"])
P3(["Probing Model C"])

subgraph PM["Frozen pretrained model"]
    direction RL

    L1{{"Module 1"}} -->
    L2{{"Module 2"}} -->
    L3{{"Module 3"}} -->
    L4{{"Module 4"}} -->
    L5{{"Module 5"}} -->
    L6{{"Module 6"}} -->
    L7{{"..."}} -->
    L8{{"Module N"}}
end

L2 -..-> P1;
L3 -..-> P2;
L6 -..-> P3;

style PM fill:none,stroke:#BBB,color:white;

classDef clsProbed fill:#5485D8,stroke:#103A83,stroke-width:2px;
class L2,L3,L6 clsProbed;

classDef clsProber fill:#A1425B,stroke:#AC6C7D,stroke-width:2px,color:white;
class P1,P2,P3 clsProber;
```

The rationale behind this method is that, since the probing task is related to the distribution of interest, that means the performance of the probing model is limited by how well the pretrained model can embed information about that distribution. In particular, if the probing model can perform well in a probing task, then the pretrained model had success in encoding relevant information regarding the target domain, as its activations were the only source of information received during training.

---

## Table of Contents
1. [About this framework](#about-this-framework)
1. [Installation](#installation)
2. [Usage examples](#usage-examples)
    1. [Step-by-step example](#step-by-step-example)
    2. [Huggingface transformers example](#huggingface-transformers-example)
    3. [More examples](#more-examples)
3. [Optimizing training with pruning](#optimizing-training-with-pruning)
4. [Preconfigured probing tasks](#preconfigured-probing-tasks)
5. [References](#references)
6. [License](#license)
7. [Citation](#citation)

---

## About this framework

This frameworks provides all resources needed to train and validate your probing task. We support pretrained [PyTorch](https://pytorch.org/) (`torch.nn.Module`) and [Huggingface Transformer](https://huggingface.co/docs/transformers/index) models. In order to properly set up and train your probing tasks, you'll need:

- A custom probing dataset (preconfigured probing tasks is a work in progress);
- A pretrained model to probe; and
- A probing model architecture in mind (we provide some simple suggestions).

With these basic ingredients, the setup using Curiosity is as follows:

1. Load a pretrained model;
2. Load a probing dataset in dataloaders (`torch.utils.data.DataLoader`);
3. Set up a Task (Dataloaders + loss function + validation metrics) with `curiosidade.ProbingTaskCustom`;
4. Set up your Probing Model (any `torch.nn.Module` will do);
5. Combine your Probing Model and your Task with a `ProbingModelFactory`;
6. Attach probers into your pretrained model by using `ProbingModelFactory.attach_probers`;
7. Train your probing models; and,
8. Check probing results, and done.

Check [usage examples in this README](#usage-examples) and also [example notebooks](./examples) for concrete examples on how everything works.

```{mermaid}
  :align: center
flowchart TB

L2{{"Frozen pretrained module"}}

subgraph PMF["ProbingTaskFactory"]
    direction LR
    
    subgraph TASK["Probing Task"]
        D1["Probing DataLoader: train"]
        D2["Probing DataLoader: eval"]
        D3["Probing DataLoader: test"]
        LF["Loss function"]
        MT["Validation metrics"]
    end

    P1(["Probing Model"])
    TASK(["Task"])
    OPT(["Optimizer"])

    P1 o==o TASK o==o OPT o==o P1
end

P1 --> Metrics["Validation metric scores"];

L2 -.-> P1;

style OPT fill:#376E56;
style LF fill:#7F4B52;
style MT fill:#5E6B3D;
style Metrics fill:#5B5531;

classDef default color:white;

classDef clsDataloader fill:#4D2E4C,stroke:#BBB;
class D1,D2,D3 clsDataloader;

classDef clsContainer fill:none,stroke:#BBB;
class PMF,TASK clsContainer;

classDef clsProbed fill:#354675,stroke:#103A83,stroke-width:2px;
class L2,L3,L6 clsProbed;

classDef clsProber fill:#A1425B,stroke:#AC6C7D,stroke-width:2px;
class P1,P2,P3 clsProber;
```

---

## Installation
```shell
python -m pip install "git+https://github.com/ulysses-camara/ulysses-curiosity"
```

To install additional dependencies, needed to run notebook examples:
```shell
python -m pip install "ulysses-curiosity[examples] @ git+https://github.com/ulysses-camara/ulysses-curiosity"
```

To install developer dependencies:
```shell
python -m pip install "ulysses-curiosity[dev] @ git+https://github.com/ulysses-camara/ulysses-curiosity"
```

---

## Usage examples

### Step-by-step example
To train probing models with Curiosity, there are a few steps that you need to follow. First, it is required to load your pretrained model and set up a probing model that inherits from the `torch.nn.Module`. The probing model must receive as its first `__init__` argument its input dimension (an integer), and the output dimension as the second argument (also an integer): `__init__(self, input_dim: int, output_dim: int)`. Note that the input dimension of any probing model vary accordingly to the output dimension of the probed module that it is attached to, and its output dimensions depends on the nature of the probing task.
```python
# (1): import needed packages.
import curiosidade
import torch
import torch.nn
import numpy as np

# (2): load your pretrained model.
pretrained_model = load_my_pretrained_model(...)

# (3): set up your probing model.
class ProbingModel(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.params = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 20),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(20, output_dim),
        )
    
    def forward(self, X):
        return self.params(X)
```

For convenience, the probing model shown above can be created with a utility function available in this package:

```python
ProbingModel = curiosidade.probers.utils.get_probing_model_feedforward(hidden_layer_dims=[20])
```

The next step is to create a probing task. This step set up aspects regarding the training, such as the train, evaluation, and test dataloaders, the loss function, and the validation metrics collected during training. Note that *evaluation and test dataloaders are optional, but recommended*. The example below show a probing task with 3 classes, using Cross Entropy as loss function, and collecting Accuracy and F1 Scores per batch. *The loss value for every batch is always collected automatically*, and will be the only metric recorded if you don't provide any.

```python
import torchmetrics

num_classes = 3

# Note: here we are using 'torchmetrics' as a suggestion, but you can use whatever you like.
accuracy_fn = torchmetrics.Accuracy(num_classes=num_classes).to("cpu")
f1_fn = torchmetrics.F1Score(num_classes=num_classes).to("cpu")

def metrics_fn(logits: torch.Tensor, truth_labels: torch.Tensor) -> dict[str, float]:
    accuracy = accuracy_fn(logits, truth_labels).detach().cpu().item()
    f1 = f1_fn(logits, truth_labels).detach().cpu().item()
    return {"accuracy": accuracy, "f1": f1}


probing_dataloader_train = torch.utils.data.DataLoader(..., shuffle=True)
probing_dataloader_eval = torch.utils.data.DataLoader(..., shuffle=False)
probing_dataloader_test = torch.utils.data.DataLoader(..., shuffle=False)


# (4): set up your probing task.
task = curiosidade.ProbingTaskCustom(
    probing_dataloader_train=probing_dataloader_train,
    probing_dataloader_eval=probing_dataloader_eval,
    probing_dataloader_test=probing_dataloader_test,
    loss_fn=torch.nn.CrossEntropyLoss(),
    task_name="debug_task",
    task_type="classification",
    output_dim=num_classes,  # Note: set to '1' if binary classification.
    metrics_fn=metrics_fn,
)
```

Now, we need to attach the probing models to the pretrained model modules. This is achieved through a `ProbingModelFactory` and its method `.attach_probers(...)`. In this step you also need to provide the optimizer responsible to updating the probing model weights. The probing model and the optimizer *should not be instantiated* when provided to the ProbingModelFactory. The factory will create brand-new copies for each probed module. The probed modules are specified either explicitly by their names, or by regular expression patterns that matches their names. Note: you can list module names by using PyTorch's built-in `pretrained_model.named_modules()`:

```python
pretrained_modules_available_for_probing = [
    name for name, _ in pretrained_model.named_modules() if name
]
print(pretrained_modules_available_for_probing)
```

```python
import functools

# (5): set up a ProbingModelFactory, which combines the probing model and the probing task.
probing_factory = curiosidade.ProbingModelFactory(
    probing_model_fn=ProbingModel,  # Note: do not instantiate.
    optim_fn=functools.partial(torch.optim.Adam, lr=0.001),  # Note: do not instantiate.
    task=task,
)

# (6): attach the probing models to the pretrained model layers.
prober_container = curiosidade.attach_probers(
    base_model=pretrained_model,
    probing_model_factory=probing_factory,
    modules_to_attach="params.relu\d+",  # or a container like ["name_a", "name_b", ...]
    random_seed=16,
    device="cpu",
)

print(f"{prober_container = }")  # Configuration summary.
print(f"{prober_container.probed_modules = }")  # Lists all probed module names.
```

By default, during attachment the input dimension of each probing model is inferred by forwarding a sample batch in the pretrained model. You can specify the input dimension explicitly by using the argument `modules_input_dim` of `attach_probers` method if necessary, as depicted in the example below. Dimensions not explicitly provided will still be inferred, so you can list only modules that are causing you trouble.

```python
prober_container = curiosidade.attach_probers(
    ...,
    modules_input_dim={
      "params.relu1": 25,
      "params.relu2": 35,
    },
)
```

Now we are set up. To train our attached probing models:

```python
# (7): train probing models.
probing_results = prober_container.train(num_epochs=5)
```

Finally, the results can be aggregated for better analysis, and visualization:

```python
# (8): aggregate results.
df_train, df_eval, df_test = probing_results.to_pandas(
    aggregate_by=["batch_index"],
    aggregate_fn=[np.mean, np.std],
)

print(df_train)
#    epoch        module metric_name    metric          
#                                         mean       std
# 0      0  params.relu1    accuracy  0.330556  0.158239
# 1      0  params.relu1          f1  0.330556  0.158239
# 2      0  params.relu1        loss  0.732363  0.050406
# 3      0  params.relu2    accuracy  0.900000  0.147358
# 4      0  params.relu2          f1  0.900000  0.147358
# 5      0  params.relu2        loss  0.531079  0.072262
# 6      1  params.relu1    accuracy  0.577778  0.226919
# ...
# 24     4  params.relu1    accuracy  0.952778  0.060880
# 25     4  params.relu1          f1  0.952778  0.060880
# 26     4  params.relu1        loss  0.399986  0.045514
# 27     4  params.relu2    accuracy  0.991667  0.028031
# 28     4  params.relu2          f1  0.991667  0.028031
# 29     4  params.relu2        loss  0.064271  0.030769
```

### Huggingface transformers example
The process shown in the previous example should be pretty much universal for any `torch.nn.Module`, which includes Huggingface's transformers (in PyTorch format). We will repeat the procedure shown previously to probe a pretrained BERT model for token classification, but this time a little bit faster with the details:
```python
# (1): import needed packages.
import functools

import curiosidade
import torch
import torch.nn
import transformers
import numpy as np


# (2): load your pretrained model.
bert = transformers.BertForTokenClassification.from_pretrained("<model-uri>")

# (3): set up your probing model.
class ProbingModel(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.params = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, output_dim),
        )
    
    def forward(self, X):
        out = X  # shape: (batch_size, largest_sequence_length, embed_dim=input_dim)
        out, _ = out.max(axis=1)  # shape: (batch_size, embed_dim=input_dim)
        out = self.params(out)  # shape: (batch_size, output_dim)
        return out


# Or, using an available utility function:
ProbingModel = curiosidade.probers.utils.get_probing_model_for_sequences(
    hidden_layer_dims=[128],
    pooling_strategy="max",
    pooling_axis=1,
)


def accuracy_fn(logits, target):
    _, cls_ids = logits.max(axis=-1)
    return {"accuracy": (cls_ids == target).float().mean().item()}


probing_dataloader_train = torch.utils.data.DataLoader(..., shuffle=True)
probing_dataloader_eval = torch.utils.data.DataLoader(..., shuffle=False)
probing_dataloader_test = torch.utils.data.DataLoader(..., shuffle=False)
num_classes: int = ...


# (4): set up your probing task.
task = curiosidade.ProbingTaskCustom(
    probing_dataloader_train=probing_dataloader_train,
    probing_dataloader_eval=probing_dataloader_eval,
    probing_dataloader_test=probing_dataloader_test,
    loss_fn=torch.nn.CrossEntropyLoss(),
    task_name="debug_task_bert",
    output_dim=num_classes,
    metrics_fn=accuracy_fn,
)

# (5): set up a ProbingModelFactory, which combines the probing model and the probing task.
probing_factory = curiosidade.ProbingModelFactory(
    task=task,
    probing_model_fn=ProbingModel,
    optim_fn=functools.partial(torch.optim.Adam, lr=0.005),
)

# (6): attach the probing models to the pretrained model layers.
prober_container = curiosidade.attach_probers(
    base_model=bert,
    probing_model_factory=probing_factory,
    modules_to_attach="bert.encoder.layer.\d+.output.dense",
    device="cuda",
)

# (7): train probing models.
probing_results = prober_container.train(num_epochs=10, show_progress_bar="epoch")

# (8): aggregate results.
df_train, df_eval, df_test = probing_results.to_pandas(
    aggregate_by=["batch_index"],
    aggregate_fn=[np.mean, np.std],
)
```

### More examples
You can find notebooks showcasing more examples in the [Examples](./examples) directory.

---

## Optimizing training with pruning

There is not point in computing all pretrained model activations past the "farthest" probing model during training. When probing models does not depends on the final output of your pretrained model, pruning away unnecessary modules will reduce computational waste.

```{mermaid}
  :align: center
flowchart TB

L1{{"Pretrained module 1"}} -->
L2{{"Pretrained module 2"}} -->
L3{{"Pretrained module 3"}} -->
L4{{"Pretrained module 4"}} -->
L5{{"..."}} -->
L6{{"Pretrained module N"}}


P1(["Probing Model"])

L2 -....-> P1;
L3 & L4 & L5 & L6 --- Waste(("Wasted\ncomputation"))


classDef default fill:#222,color:white;

style Waste stroke-style:dashed,stroke-dasharray:8,color:black;

classDef clsProbed fill:#354675,stroke:#103A83,stroke-width:2px;
class L2 clsProbed;

classDef clsProber fill:#A1425B,stroke:#AC6C7D,stroke-width:2px;
class P1 clsProber;

classDef clsWaste fill:#D82626,stroke:#103A83,stroke-width:2px;
class L3,L4,L5,L6 clsWaste;
```

This optimization is activated upon calling `attach_probers`, as exemplified below:

```python
prober_container = curiosidade.attach_probers(
    ...,
    prune_unrelated_modules="infer",
)
```

When `prune_unrelated_modules="infer"`, the forward flow will be interrupted right after the last probed module ends its forward. This strategy should work for any model that has a deterministic forward flow. If this is not working for your architecture, you can provide a list of module names to prune, as exemplified below:

```python
prober_container = curiosidade.attach_probers(
    ...,
    prune_unrelated_modules=["module_name_a"],
)
```
Be careful to not prune a required module, or else a `RuntimeError` should be raised during training. Since the forward flow will never get past through a pruned module, you do not need to list any module past the first pruned moudle.

---

## Preconfigured probing tasks

This package provides a collection of preconfigured probing tasks for PT-br language following the descriptions in [(Conneau et. al, 2018)](https://aclanthology.org/P18-1198/). Preconfigured tasks enable easier probing setup, and standard train, evaluation and test data splits.

To programatically list all available preconfigured probing tasks:
```python
probing_tasks = curiosidade.get_available_preconfigured_tasks()
# Return format: sequence of (class name, class) pairs
```

The following preconfigured probing tasks are available in this package:
```python
curiosidade.ProbingTaskSentenceLength
curiosidade.ProbingTaskWordContent
curiosidade.ProbingTaskBigramShift
curiosidade.ProbingTaskTreeDepth
curiosidade.ProbingTaskTopConstituent
curiosidade.ProbingTaskPastPresent
curiosidade.ProbingTaskSubjectNumber
curiosidade.ProbingTaskObjectNumber
curiosidade.ProbingTaskSOMO
curiosidade.ProbingTaskCoordinationInversion
```

All preconfigured probing tasks follows a standard `__init__` API.

```python
import typing as t

def fn_text_to_tensor_for_pytorch(
    content: list[str],
    labels: list[int],
    split: t.Literal["train", "eval", "test"],
) -> dict[str, torch.Tensor]:
    """Transform raw text data into a PyTorch dataloader."""
    X = torch.nn.utils.rnn.pad_sequence(
        [torch.Tensor(inst.ids)[:32] for inst in tokenizer.encode_batch(content)],
        batch_first=True,
        padding_value=0.0,
    )

    y = torch.Tensor(labels)

    X = X.long()
    y = y.long()

    return torch.utils.data.TensorDataset(X, y)


def metrics_fn(logits, target):
    _, cls_ids = logits.max(axis=-1)
    return {"accuracy": (cls_ids == target).float().mean().item()}


task = curiosidade.ProbingTaskSentenceLength(
    fn_raw_data_to_tensor=fn_text_to_tensor_for_pytorch,
    metrics_fn=metrics_fn,
    data_domain="general-pt-br",  # PT-br wikipedia probing dataset
    output_dir="probing_datasets",  # Directory to store downloaded probing datasets
    batch_size_train=128,
    batch_size_eval=256,
)

# Analogously to all other available preconfigured probing tasks.
```

You can check [a full example notebook](./examples/6_using_a_preconfigured_probing_task.ipynb) showcasing the use of preconfigured probing tasks.

---

## References
[Alexis Conneau, German Kruszewski, Guillaume Lample, Loïc Barrault, and Marco Baroni. 2018. What you can cram into a single $&!#\* vector: Probing sentence embeddings for linguistic properties. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2126–2136, Melbourne, Australia. Association for Computational Linguistics.](https://aclanthology.org/P18-1198/)

---

## License
```markdown
MIT License

Copyright (c) 2022 Felipe Alves Siqueira

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```


---

## Citation
```bibtex
@inproceedings{
    author="",
    date="",
}
```
