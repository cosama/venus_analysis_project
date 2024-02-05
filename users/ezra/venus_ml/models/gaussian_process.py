from typing import Optional, Callable

from numpy import array
from sklearn.gaussian_process import GaussianProcessRegressor

from .wrappers import SklearnWrapper


class GaussianProcess(SklearnWrapper):
    """
    Gaussian Process wrapped in SklearnWrapper
    """
    def __init__(self, hyperparameters: dict, evaluation_metric: Optional[Callable[[array, array], float]] = None):
        """
        Initializes a wrapped Gaussian Process Regressor
        Args:
            hyperparameters (dict): hyperparameters used to define the GPR model
            evaluation_metric (Callable): metric used to evaluate the model
        """
        model = GaussianProcessRegressor(hyperparameters)
        super().__init__(model, evaluation_metric)