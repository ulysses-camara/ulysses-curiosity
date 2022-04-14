import torch

from . import base


class TaskSentenceLength(base.BaseTask):
    pass


class TaskWordContent(base.BaseTask):
    pass


class TaskBigramShift(base.BaseTask):
    pass


class TaskTreeDepth(base.BaseTask):
    pass


class TaskTopConstituent(base.BaseTask):
    pass


class TaskTense(base.BaseTask):
    pass


class TaskSubjectNumber(base.BaseTask):
    pass


class TaskObjectNumber(base.BaseTask):
    pass


class TaskSOMO(base.BaseTask):
    pass


class TaskCoordinationInversion(base.BaseTask):
    pass


class TaskCustom(base.BaseTask):
    def __init__(
        self,
        probing_dataloader: torch.utils.data.DataLoader,
        loss_fn,
        task_name: str = "unnamed_task",
    ):
        super().__init__(
            probing_dataloader=probing_dataloader,
            loss_fn=loss_fn,
            task_name=task_name,
        )
