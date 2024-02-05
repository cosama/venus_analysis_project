from typing import Optional, Callable

from numpy import array
from sklearn.ensemble import RandomForestRegressor

from .wrappers import SklearnWrapper


class RandomForest(SklearnWrapper):
    """
    Random Forest Regressor wrapped in SklearnWrapper
    """
    def __init__(self, hyperparameters: dict, evaluation_metric: Optional[Callable[[array, array], float]] = None):
        """
        Initializes a wrapped Random Forest
        Args:
            hyperparameters (dict): hyperparameters to define the random forest model
            evaluation_metric (Callable): metric used to evaluate the model
        """
        model = RandomForestRegressor(**hyperparameters)
        super().__init__(model, evaluation_metric)
