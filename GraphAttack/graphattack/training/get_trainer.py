from typing import Union

import torch

from graphattack import training


def get_trainer(model: Union[str, torch.nn.Module]) -> training.trainer.Trainer:
    """get the default trainer using str or model

    Parameters
    ----------
    model : Union[str, torch.nn.Module]
        the model to be trained in

    Returns
    -------
    graphattack.training.trainer.Trainer
        the default trainer for the model.

    Examples
    --------
    >>> import graphattack
    >>> graphattack.training.get_trainer('GCN')
    graphattack.training.trainer.Trainer

    >>> from graphattack.models import GCN
    >>> graphattack.training.get_trainer(GCN)
    graphattack.training.trainer.Trainer

    >>> # by default, it returns `graphattack.training.Trainer`
    >>> graphattack.training.get_trainer('unimplemeted_model')
    graphattack.training.trainer.Trainer

    >>> graphattack.training.get_trainer('RobustGCN')
    graphattack.training.robustgcn_trainer.RobustGCNTrainer

    >>> # it is case-sensitive
    >>> graphattack.training.get_trainer('robustGCN')
    graphattack.training.trainer.Trainer
    """
    default = training.Trainer
    if isinstance(model, str):
        class_name = model
    else:
        class_name = model.__class__.__name__

    trainer = getattr(training, model + "Trainer", default)
    return trainer
