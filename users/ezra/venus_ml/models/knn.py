from typing import Optional, Callable

from numpy import array
from sklearn.neighbors import KNeighborsRegressor

from .wrappers import SklearnWrapper


class KNN(SklearnWrapper):
    """
    K-Nearest Neighbors Regressor wrapped in SklearnWrapper
    """
    def __init__(self, hyperparameters: dict, evaluation_metric: Optional[Callable[[array, array], float]] = None):
        """
        Initializes a wrapped KNN Regressor
        Args:
            hyperparameters (dict): hyperparameters used to define the KNN Regressor
            evaluation_metric (Callable): metric used to evaluate the model
        """
        model = KNeighborsRegressor(**hyperparameters)
        super().__init__(model, evaluation_metric)
