import math

import os

import numpy as np
from sklearn.neighbors import KDTree
from tqdm import tqdm
import xgboost


class Base_Regressor:
    def k_fold_cross_validation(self, x_df_list, y_df_list, parquet_files):
        # make lists to store all the data
        label_list = []
        mse_list = []
        mae_list = []
        mape_list = []
        n_list = []

        # K fold on each of the dataframes
        for i, _ in enumerate(x_df_list):
            # Get testing data
            x_test = x_df_list[i]
            y_test = y_df_list[i]

            # Get training data
            x_train = np.concatenate(x_df_list[:i] + x_df_list[i + 1:])
            y_train = np.concatenate(y_df_list[:i] + y_df_list[i + 1:])

            # fit self
            self.fit(x_train, y_train)

            # Predict on the test data with a progress bar
            y_pred = np.zeros_like(y_test)
            y_len = len(y_pred)
            for j, x in tqdm(enumerate(x_test), total=y_len, desc="Predicting"):
                new_x = x.reshape(1, -1)
                y_pred[j] = self.predict(new_x)

            # Calculate mean losses
            mse_loss = np.mean((y_pred - y_test) ** 2)
            mae_loss = np.mean(np.abs(y_pred - y_test))
            mape_loss = np.mean(np.abs((y_pred - y_test) / y_test)) * 100

            label_list.append(os.path.basename(parquet_files[i]))
            mse_list.append(mse_loss)
            mae_list.append(mae_loss)
            mape_list.append(mape_loss)
            n_list.append(y_len)

        return label_list, mse_list, mae_list, mape_list, n_list



class KNN_Regressor(Base_Regressor):
    def __init__(self, to_normalize, num_neighbors, weights):
        self.num_neighbors = num_neighbors
        self.to_normalize = to_normalize
        self.maximum = None
        self.minimum = None
        self.weights = weights


    def fit(self, x_data, y_data):
        if None == self.maximum:
            self.maximum = np.max(x_data)
        else:
            self.maximum = np.max(self.maximum, np.max(x_data))

        if None == self.minimum:
            self.minimum = np.min(x_data)
        else:
            self.minimum = np.min(self.minimum, np.min(x_data))

        self.kd_tree = KDTree(self.weigh(self.normalize(x_data)))
        self.y_data = y_data


    def normalize(self, data):
        if self.to_normalize:
            data = (data - self.minimum) / (self.maximum - self.minimum)
        return (data)


    def weigh(self, data):
        # we want certain dimensions to be more imporant than others
        # so we weigh them
        return self.weights * data


    def predict(self, data):
        _, indices = self.kd_tree.query(self.weigh(self.normalize(data)).reshape(1, -1), k=self.num_neighbors)
        k_nearest_labels = self.y_data[indices.flatten()]
        return np.mean(k_nearest_labels)



class Tree_Regressor(Base_Regressor):
    def __init__(self, **kwargs):
        self.model = xgboost.XGBRegressor(**kwargs)


    def fit(self, x_data, y_data):
        self.model.fit(x_data, y_data)


    def predict(self, data):
        return self.model.predict(data)



class Cheat_Regressor(Base_Regressor):
    def __init__(self):
        self.pred_dict = {}
        self.point_column = 0


    def fit(self, x_data, y_data):
        bin_values = x_data[:, self.point_column]
        unique_bins = np.unique(bin_values)
        for bin_value in unique_bins:
            bin_indices = np.where(bin_values == bin_value)
            bin_y_data = y_data[bin_indices]
            self.pred_dict[bin_value] = np.mean(bin_y_data)


    def predict(self, data):
        return self.pred_dict[data[0][self.point_column]]


    def k_fold_cross_validation(self, x_df_list, y_df_list, parquet_files):
        # make lists to store all the data
        label_list = []
        mse_list = []
        mae_list = []
        mape_list = []
        n_list = []

        # K fold on each of the dataframes
        for i, _ in enumerate(x_df_list):
            # Get testing data
            x_test = x_df_list[i]
            y_test = y_df_list[i]

            # Get training data
            x_train = np.concatenate(x_df_list[:i] + x_df_list[i + 1:])
            y_train = np.concatenate(y_df_list[:i] + y_df_list[i + 1:])

            # fit self
            self.fit(x_test, y_test)

            # Predict on the test data with a progress bar
            y_pred = np.zeros_like(y_test)
            y_len = len(y_pred)
            for j, x in tqdm(enumerate(x_test), total=y_len, desc="Predicting"):
                new_x = x.reshape(1, -1)
                y_pred[j] = self.predict(new_x)

            # Calculate mean losses
            mse_loss = np.mean((y_pred - y_test) ** 2)
            mae_loss = np.mean(np.abs(y_pred - y_test))
            mape_loss = np.mean(np.abs((y_pred - y_test) / y_test)) * 100

            label_list.append(os.path.basename(parquet_files[i]))
            mse_list.append(mse_loss)
            mae_list.append(mae_loss)
            mape_list.append(mape_loss)
            n_list.append(y_len)

        return label_list, mse_list, mae_list, mape_list, n_list

