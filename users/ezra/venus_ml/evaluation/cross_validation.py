from typing import Sequence, Union, Tuple, List

import pandas as pd

from ..data import VenusDataset


class EvaluationManager:
    """
    Manager class to manage evaluation of a set of models
    Attributes:
        models (Sequence[Union[PyTorchWrapper, SklearnWrapper]]): Sequence of wrapped models
        dataset_config (dict): Dictionary of dataset initialization parameters
        cv_results (list): List of dictionaries with evaluation results for each model and step of cross validation
    """
    def __init__(self, models: Sequence, dataset_config):
        """
        Initialize the evaluation manager
        Args:
            models (Sequence): Sequence of wrapped models
            dataset_config (dict): Dictionary of dataset initialization parameters
        """
        self.models = models
        self.dataset_config = dataset_config
        self.cv_results = []

    def split_dataset(self, train_runs: Union[int, float, Sequence[float]],
                      validation_runs: Union[int, float, Sequence[float]]) -> Tuple[VenusDataset, VenusDataset]:
        """
        Split the dataset into training and validation sets for use in cross validation
        Args:
            train_runs (Union[int, float, Sequence[float]]): Training set runs
            validation_runs (Union[int, float, Sequence[float]]): Validation set runs

        Returns:
            A tuple containing training/validation datasets
        """
        train_config = self.dataset_config.copy()
        val_config = self.dataset_config.copy()

        train_config["run_selection"] = train_runs
        val_config["run_selection"] = validation_runs
        return VenusDataset(**train_config), VenusDataset(**val_config)

    def train_all(self, dataset: VenusDataset) -> List[float]:
        """
        Trains each model on the given dataset
        Args:
            dataset: The given dataset

        Returns:
            Evaluation metric for training set
        """
        for model in self.models:
            model.train(dataset)
        return self.evaluate_all(dataset)

    def evaluate_all(self, dataset: VenusDataset) -> List[float]:
        """
        Evaluates each model on the given dataset and returns evaluation metrics
        Args:
            dataset: dataset to evaluate the models on
        Returns:
            Evaluation metric for the given dataset
        """
        errors = []
        for model in self.models:
            errors.append(model.evaluate(dataset))
        return errors

    def k_fold_cv(self, runs=(5.0, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0)):
        """
        Leave one out cross validation with the given set of runs. Results from each round are saved to self.cv_results
        and can be serialized to a .csv file
        Args:
            runs: Which runs to use from the Venus dataset
        """
        for i in range(len(runs)):
            run_list = list(runs)
            validation_run = run_list.pop(i)
            train_runs = run_list

            train_dataset, val_dataset = self.split_dataset(train_runs, validation_run)
            train_errors = self.train_all(train_dataset)
            validation_errors = self.evaluate_all(val_dataset)
            for train_error, validation_error, model in zip(train_errors, validation_errors, self.models):
                self.cv_results.append({
                    "validation_run": validation_run,
                    "model": type(model).__name__,
                    "train_error": train_error,
                    "validation_error": validation_error,
                })

    def save_results(self, file_path):
        """
        Save results to a .csv file
        Args:
            file_path: Path to the file
        """
        results_df = pd.DataFrame(self.cv_results)
        results_df.to_csv(file_path, index=False)
