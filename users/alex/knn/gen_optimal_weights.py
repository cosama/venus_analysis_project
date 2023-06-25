import argparse

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from tqdm import tqdm

import knn



def read_csv_files(csv_files, column_names):
    dfs = []
    for csv_file in csv_files:
        dfs.append(pd.read_csv(csv_file, names=column_names, delimiter=" "))
    return dfs



def calculate_losses(y_pred, y_test):
    mse_loss = np.mean((y_pred - y_test) ** 2)
    mae_loss = np.mean(np.abs(y_pred - y_test))
    mape_loss = np.mean(np.abs((y_pred - y_test) / y_test)) * 100
    return mse_loss, mae_loss, mape_loss



def predict_test_data(knn_model, x_test):
    y_pred = np.zeros((len(x_test), knn_model.y_data.shape[1]))
    for i, x in tqdm(enumerate(x_test), total=len(x_test), desc="Predicting"):
        y_pred[i] = knn_model.predict(x)
    return y_pred



def train_and_test_knn(csv_files, column_names, predict_columns, num_epochs):
    mse_losses = []
    mae_losses = []
    mape_losses = []
    datapoints = []

    best_loss = float('inf')
    best_weights = np.ones(column_names.shape[0] - len(predict_columns))

    dataframes = read_csv_files(csv_files, column_names)
    x_dataframes = list(map(lambda x: x.drop(predict_columns, axis=1).values, dataframes))
    y_dataframes = list(map(lambda x: x[predict_columns].values, dataframes))


    for _ in range(num_epochs):
        total_loss = 0.0
        for i, test_dataframe in enumerate(dataframes):
            x_train_dataframes = dataframes[:i] + dataframes[i + 1:]
            y_train_dataframes = dataframes[:i] + dataframes[i + 1:]

            # Read training data
            x_train = pd.concat(x_train_dataframes)
            y_train = pd.concat(y_train_dataframes)

            # Read testing data
            x_test = x_dataframes[i]
            y_test = y_dataframes[i]

            # Generate random weights
            weights = np.abs(best_weights + np.random.normal(0, 10, size=best_weights.shape))

            # Train the KNN regression model with random weights
            knn_model = knn.KNNRegressionModel(x_train, y_train, weights=weights)

            # Predict on the test data
            y_pred = predict_test_data(knn_model, x_test)

            # Calculate losses
            mse_loss, mae_loss, mape_loss = calculate_losses(y_pred, y_test)

            total_loss += mape_loss * len(y_test)

        print(total_loss, best_loss)
        # Update best weights if loss is improved
        if total_loss < best_loss:
            best_loss = total_loss
            best_weights = weights
            print(best_weights)

    for i, test_dataframe in enumerate(dataframes):
        x_train_dataframes = dataframes[:i] + dataframes[i + 1:]
        y_train_dataframes = dataframes[:i] + dataframes[i + 1:]

        # Read training data
        x_train = pd.concat(x_train_dataframes)
        y_train = pd.concat(y_train_dataframes)

        # Read testing data
        x_test = x_dataframes[i]
        y_test = y_dataframes[i]

        # Train the KNN regression model with the best weights
        knn_model = knn.KNNRegressionModel(x_train, y_train, weights=best_weights)

        # Predict on the test data with the best model
        y_pred = predict_test_data(knn_model, x_test)

        # Calculate losses
        mse_loss, mae_loss, mape_loss = calculate_losses(y_pred, y_test)
        mse_losses.append(mse_loss * len(y_test))
        mae_losses.append(mae_loss * len(y_test))
        mape_losses.append(mape_loss * len(y_test))
        datapoints.append(len(y_test))

        # Print individual losses for each CSV file
        print(f"Loss for CSV {i + 1}:")
        print("Mean Squared Error (MSE) Loss:", mse_loss)
        print("Mean Absolute Error (MAE) Loss:", mae_loss)
        print("Mean Absolute Percentage Error (MAPE) Loss:", mape_loss)
        print()

    return mse_losses, mae_losses, mape_loss, datapoints



def calculate_weighted_average(losses, datapoints):
    weighted_avg_loss = np.sum(losses) / np.sum(datapoints)
    return weighted_avg_loss



if __name__ == "__main__":
    data_dir = "../../../clean_data/vary_pressure_and_bias_voltage/"

    parser = argparse.ArgumentParser(description="KNN Regression")
    parser.add_argument("--csv_files", required=True, nargs="+", help="CSV data files")
    parser.add_argument("--column_names_file", default="column_names", help="Column names CSV file")
    parser.add_argument("--predict_columns", default=["fcv1_i"], nargs="+", help="Columns to predict")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs for random search")
    args = parser.parse_args()

    csv_files = list(map(lambda x: data_dir + x, args.csv_files))
    column_names_file = data_dir + args.column_names_file
    predict_columns = args.predict_columns
    num_epochs = args.num_epochs

    # Read column names
    column_names = np.array(list(pd.read_csv(column_names_file, delimiter=", ", engine='python')))

    # Train and test KNN regression
    mse_losses, mae_losses, mape_losses, datapoints = train_and_test_knn(csv_files, column_names, predict_columns, num_epochs)

    # Calculate weighted average losses
    weighted_avg_mse_loss = calculate_weighted_average(mse_losses, datapoints)
    weighted_avg_mae_loss = calculate_weighted_average(mae_losses, datapoints)
    weighted_avg_mape_loss = calculate_weighted_average(mape_losses, datapoints)

    # Print weighted average losses
    print("Weighted Average Losses:")
    print("Mean Squared Error (MSE) Loss:", weighted_avg_mse_loss)
    print("Mean Absolute Error (MAE) Loss:", weighted_avg_mae_loss)
    print("Mean Absolute Percentage Error (MAPE) Loss:", weighted_avg_mape_loss)

