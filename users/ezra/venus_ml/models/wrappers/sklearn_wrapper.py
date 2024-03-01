from typing import Optional, Callable

import torch
from numpy import array
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error

class SklearnWrapper:
    """
        Wrapper class for scikit-learn models to standardize training, prediction, and evaluation.

        Attributes:
            eval_metric (callable, optional): Function used in evaluation
            model (BaseEstimator): The scikit-learn model to be wrapped.
             framework (str): The framework being wrapped, used to check wrapper type

    """
    def __init__(self, model: BaseEstimator, eval_metric: Optional[Callable[[array, array], float]] = None):
        """
        Initializes the SklearnModelWrapper instance.

        Args:
            model (BaseEstimator): The scikit-learn model to be wrapped.
        """
        self.eval_metric = eval_metric if eval_metric is not None else mean_squared_error
        self.model = model
        self.framework = "sklearn"

    def predict(self, data: torch.tensor) -> torch.tensor:
        """
            Generates predictions using the model.

            Args:
                data (torch.tensor): Data features for making predictions.

            Returns:
                torch.tensor: Predictions made by the model.
        """
        if type(data) is torch.Tensor:
            data = data.numpy()
        return torch.tensor(self.model.predict(data))

    def evaluate(self, inputs, outputs) -> float:
        """
            Evaluates the model's performance on a given set of input/output pairs

            Args:
                 inputs (array): Testing data
                outputs (array): Testing targets

            Returns:
                float: The evaluation metric of the model on a dataset
        """
        assert inputs.shape[0] == outputs.shape[0], "Number of inputs samples and outputs samples do not match"
        predictions = self.predict(inputs)
        return self.eval_metric(outputs, predictions)

    def fit(self, inputs, outputs):
        """
            Trains the model using the provided training data.

            Args:
                inputs (array): Training data
                outputs (array): Training targets
        """
        self.model.fit(inputs, outputs.ravel())
