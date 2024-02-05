from typing import Any, Dict, List

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skorch import NeuralNetRegressor


# TODO think about if the CV method works


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
            self.model = NeuralNetRegressor(model_wrapper.model, device=model_wrapper.device)

        elif model_wrapper.framework == "sklearn":
            self.model = model_wrapper.model

        if search_method == "grid":
            self.search = GridSearchCV(estimator=self.model, param_grid=param_grid, verbose=verbosity)
        elif search_method == "random":
            self.search = RandomizedSearchCV(estimator=self.model, param_distributions=param_grid, verbose=verbosity)

    def tune(self, X, y):
        """
        Call fit on the search method and then print out the best model hyperparameters
        Args:
            X:
            y:

        Returns:
            A dataframe containing the results

        """
        self.search.fit(X, y)
        print("Best model hyperparameters:")
        for name, value in self.search.best_params_.items():
            print(f"{name}: {value}")
