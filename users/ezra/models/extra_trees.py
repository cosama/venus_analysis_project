from typing import Optional, Callable

from numpy import array
from sklearn.ensemble import ExtraTreesRegressor

from users.ezra.models import SklearnWrapper


class ExtraTrees(SklearnWrapper):
    """
    ExtraTreesRegressor wrapped in SklearnWrapper
    """
    def __init__(self, hyperparameters: dict, evaluation_metric: Optional[Callable[[array, array], float]] = None):
        """
        Initializes wrapped ExtraTreesRegressor
        Args:
            hyperparameters (dict): hyperparameters used to define the extra trees model
            evaluation_metric (Callable): metric used to evaluate the model
        """
        model = ExtraTreesRegressor(**hyperparameters)
        super().__init__(model, evaluation_metric)