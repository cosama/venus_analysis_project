from typing import Optional, Callable

import torch
from numpy import array
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error

from users.ezra.data import VenusDataset


class SklearnWrapper:
    """
        Wrapper class for scikit-learn models to standardize training, prediction, and evaluation.

        Attributes:
            model (BaseEstimator): The scikit-learn model to be wrapped.
    """
    def __init__(self, model: BaseEstimator, eval_metric: Optional[Callable[[array, array], float]] = None):
        """
        Initializes the SklearnModelWrapper instance.

        Args:
            model (BaseEstimator): The scikit-learn model to be wrapped.
        """
        self.eval_metric = eval_metric if eval_metric is not None else mean_squared_error
        self.model = model

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

    def evaluate(self, dataset) -> float:
        """
            Evaluates the model's performance on a dataset

            Args:
                dataset (VenusDataset): The dataset to evaluate the model on

            Returns:
                float: The evaluation metric of the model on a dataset
        """
        inputs, outputs = dataset.to_numpy()
        predictions = self.predict(inputs)
        return self.eval_metric(outputs, predictions)

    def train(self, dataset: VenusDataset, epochs: Optional[int] = 1):
        """
            Trains the model using the provided training data.

            Args:
                dataset (VenusDataset): Training data features
                epochs (int, optional): parameter to be consistent with PyTorch wrapper
        """
        inputs, outputs = dataset.to_numpy()
        self.model.fit(inputs, outputs.ravel())
