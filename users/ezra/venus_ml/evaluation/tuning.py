from typing import Any, Dict, List

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skorch import NeuralNetRegressor


class HyperparameterTuner:
    """
    Hyperparameter tuning tool that performs hyperparameter tuning via grid search or random search. Uses skorch library
    to wrap the pytorch modules and allow them to be passed into sklearn tuning methods. Uses either GridSearchCV or
    RandomizedSearchCV to perform the hyperparameter tuning. GridSearchCV is exhaustive and uses explicit values,
    while RandomizedSearchCV is random and can use distributions of hyperparameters as well.
    """

    def __init__(self, model_wrapper: Any, param_grid: Dict[str, List[Any]],
                 search_method: str = "grid", verbosity: int = 0) -> None:
        """
        Initialize the hyperparameter tuner with the given model and parameter grid
        Args:
            model_wrapper: Wrapped model
            param_grid (dict): Hyperparameter grid
            search_method (str): "grid" or "random"
            verbosity (int): Level of verbosity for the search method
        """
        if model_wrapper.framework == "torch":
            self.model = NeuralNetRegressor(model_wrapper.model, device=model_wrapper.device,
                                            verbose=0, criterion=model_wrapper.criterion)
        elif model_wrapper.framework == "sklearn":
            self.model = model_wrapper.model

        if search_method == "grid":
            self.search = GridSearchCV(estimator=self.model, param_grid=param_grid, verbose=verbosity, n_jobs=-1)
        elif search_method == "random":
            self.search = RandomizedSearchCV(estimator=self.model, param_distributions=param_grid, n_iter=100,
                                             verbose=verbosity, n_jobs=-1)
        self.searched = False

    def fit(self, X, y, verbose=False):
        """
        Call fit on the search method and then can print out the best model hyperparameters
        Args:
            X (array): Data matrix
            y (array): Target values
            verbose (bool): If True, print best model hyperparameters after search

        Returns:
            A dataframe containing the results

        """
        self.search.fit(X, y)
        self.searched = True
        if verbose:
            print(f"Best {type(self.model).__name__} hyperparameters:")
            for name, value in self.search.best_params_.items():
                print(f"{name}: {value}")

    def best_model(self):
        if self.searched:
            return self.search.best_estimator_
        print("Please call tune() to find the best estimator")

