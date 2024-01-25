from typing import Optional, Callable

from numpy import array
from sklearn.svm import SVR

from users.ezra.models import SklearnWrapper


class SVM(SklearnWrapper):
    """
    Support Vector Regressor wrapped in SklearnWrapper
    """
    def __init__(self, hyperparameters: dict, evaluation_metric: Optional[Callable[[array, array], float]] = None):
        """
        Initializes a wrapped SVM Regressor
        Args:
            hyperparameters (dict): hyperparameters used to define the SVM
            evaluation_metric (Callable): metric used to evaluate the model
        """
        model = SVR(**hyperparameters)
        super().__init__(model, evaluation_metric)