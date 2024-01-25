from typing import Sequence, Union

import pandas as pd
import torch

from users.ezra.models import PyTorchWrapper, SklearnWrapper, BaseMLP, RandomForest
from users.ezra.data.dataset import VenusDataset


class EvaluationManager:
    def __init__(self, models: Sequence[Union[PyTorchWrapper, SklearnWrapper]], dataset_config):
        self.models = models
        self.dataset_config = dataset_config
        self.cv_results = []

    def split_dataset(self, train_runs, validation_runs):
        train_config = self.dataset_config.copy()
        val_config = self.dataset_config.copy()

        train_config["run_selection"] = train_runs
        val_config["run_selection"] = validation_runs
        return VenusDataset(**train_config), VenusDataset(**val_config)

    def train_all(self, dataset, epochs=1):
        for model in self.models:
            model.train(dataset, epochs=epochs)
        return self.evaluate_all(dataset)

    def evaluate_all(self, dataset):
        errors = []
        for model in self.models:
            errors.append(model.evaluate(dataset))
        return errors

    def cross_evaluate(self, runs=(5.0, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0)):
        for i in range(len(runs)):
            run_list = list(runs)
            validation_run = run_list.pop(i)
            train_runs = run_list

            train_dataset, val_dataset = self.split_dataset(train_runs, validation_run)
            # print(train_dataset.dataset_size(), val_dataset.dataset_size())
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
        results_df = pd.DataFrame(self.cv_results)
        results_df.to_csv(file_path, index=False)
