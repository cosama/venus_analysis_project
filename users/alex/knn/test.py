import argparse

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from tqdm import tqdm

import knn


if __name__ == "__main__":
    data_dir = "../../../clean_data/vary_pressure_and_bias_voltage/"

    parser = argparse.ArgumentParser(description="KNN Regression")
    parser.add_argument("--train_csv_files", required=True, nargs="+", help="Training CSV files")
    parser.add_argument("--test_csv_file", required=True, help="Testing CSV file")
    parser.add_argument("--column_names_file", default="column_names", help="Column names CSV file")
    parser.add_argument("--predict_columns", default=["fcv1_i"], nargs="+", help="Columns to predict")
    args = parser.parse_args()

    train_csv_files = list(map(lambda x: data_dir + x, args.train_csv_files))
    test_csv_file = data_dir + args.test_csv_file
    column_names_file = data_dir + args.column_names_file
    predict_columns = args.predict_columns

    # get other columns
    column_names = list(pd.read_csv(column_names_file, delimiter=", "))
    input_columns = ["inj_mbar", "bias_v", "bias_i", "extraction_i"] + ["k18_fw", "k18_ref", "g28_fw", "puller_i", "extraction_i"] #+ ["inj_i", "ext_i", "mid_i", "sext_i", "x_ray_source", "x_ray_exit"]
    #input_columns = list(filter(lambda x: x not in predict_columns, column_names))

    # Read training data
    tmp = []
    for train_csv_file in train_csv_files:
        tmp.append(pd.read_csv(train_csv_file, names=column_names, delimiter=" "))
    train_df = pd.concat(tmp)
    x_train = train_df[input_columns].values
    y_train = train_df[predict_columns].values

    # Read test data
    test_df = pd.read_csv(test_csv_file, names=column_names, delimiter=" ")
    x_test = test_df[input_columns].values
    y_test = test_df[predict_columns].values

    # Create and train the KNN regression model
    knn_model = knn.KNNRegressionModel(x_train, y_train)

    # Predict on the test data with a progress bar
    y_pred = np.zeros_like(y_test)
    for i, x in tqdm(enumerate(x_test), total=len(x_test), desc="Predicting"):
        y_pred[i] = knn_model.predict(x)

    # Calculate mean squared error (MSE) loss
    mse_loss = np.mean((y_pred - y_test) ** 2)
    mae_loss = np.mean(np.abs(y_pred - y_test))
    mape_loss = np.mean(np.abs((y_pred - y_test) / y_test)) * 100

    print("Mean Squared Error (MSE) Loss:", mse_loss)
    print("Mean Absolute Error (MAE) Loss:", mae_loss)
    print("Mean Absolute Percentage Error (MAPE) Loss:", mape_loss)
