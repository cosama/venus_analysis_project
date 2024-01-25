from typing import Optional, Callable

from numpy import array
from sklearn.ensemble import RandomForestRegressor

from .wrappers import SklearnWrapper


class RandomForest(SklearnWrapper):
    """
    Random Forest wrapped in SklearnWrapper
    """
    def __init__(self, model_parameters: dict, evaluation_metric: Optional[Callable[[array, array], float]] = None):
        """
        Instantiates a wrapped Random Forest
        Args:
            model_parameters: parameters to define the random forest model
            evaluation_metric: metric used to evaluate the model
        """
        model = RandomForestRegressor(**model_parameters)
        super().__init__(model, evaluation_metric)
