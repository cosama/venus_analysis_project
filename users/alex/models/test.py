import argparse
import copy
import csv
import os
import re
import sys

import numpy as np
import pandas as pd
import xgboost
from tqdm import tqdm

import knn



def cli_parser():
    models = ["knn", "xgb"]
    parser = argparse.ArgumentParser(description="Test arbitrary models")

    parser.add_argument("parquet_files", nargs="+", help="Parquet files")

    subparsers = parser.add_subparsers(dest="model", required=True, help="Model to use")

    # Create subparsers so we can handle different cli
    xgb_parser = subparsers.add_parser("xgb", help="Options for xgboost regression")
    xgb_parser.add_argument("--n_estimators", required=True, type=int, help="Specify the number of estimators to use")
    xgb_parser.add_argument("--max_depth", required=True, type=int, help="Specify the maximum depth of the tree")
    xgb_parser.add_argument("--max_leaves", required=True, type=int, help="Specify the maximum number of leaf nodes to use, grows the tree in a best-first fashion")
    xgb_parser.add_argument("--max_bin", required=True, type=int, help="Specify the maximum number of bins to use for histogram-based tree methods")
    xgb_parser.add_argument("--grow_policy", required=True, choices=["depthwise", "lossguide"], help="Specify the tree growing policy, 0: favor splitting near root, 1: favor splitting at nodes with highest loss change")
    xgb_parser.add_argument("--learning_rate", required=True, type=float, help="Specify the boosting's learning rate")
    xgb_parser.add_argument("--objective", required=True, choices=["reg:squarederror", "reg:squaredlogerror", "reg:pseudohubererror", "reg:absoluteerror"], help="Specify the learning task and corresponding learning objective")
    xgb_parser.add_argument("--tree_method", required=True, choices=["exact", "hist"], help="Specify the tree construction algorithm to use")
    xgb_parser.add_argument("--reg_alpha", required=True, type=float, help="Specify the L1 regularization term on weights")
    xgb_parser.add_argument("--reg_lambda", required=True, type=float, help="Specify the L2 regularization term on weights")
    xgb_parser.add_argument("--subsample", required=True, type=float, help="Specify the fraction of random selection of rows")
    xgb_parser.add_argument("--colsample_bynode", required=True, type=float, help="Specify the fraction of columns to randomly sample at each node split")
    xgb_parser.add_argument("--colsample_bytree", required=True, type=float, help="Specify the fraction of columns to randomly sample at the creation of each tree")
    xgb_parser.add_argument("--num_boost_round", required=True, type=int, help="Specify the number of boosting iterations")
    xgb_parser.add_argument("--num_parallel_tree", required=True, type=int, help="Specify the number of trees to train in parallel")

    knn_parser = subparsers.add_parser("knn", help="Options for k nearest neighbor regression")
    knn_parser.add_argument("--num_neighbors", required=True, type=int, help="Specify the number of neighbors to use")
    knn_parser.add_argument("--to_normalize", required=True, type=int, help="Specify the whether to normalize the input data (0: False, 1: True)")

    # TODO rewrite to be modular, maybe have an extract_column_names function, but would then also have to extract a parquet file multiple times, idk, think about it
    tmp_args, _ = parser.parse_known_args()

    if 2 > len(tmp_args.parquet_files):
        raise(ValueError("parquet_files argument must have at least 2 values"))

    # dynamically generate valid column names
    column_names = list(filter(lambda x: x != "fcv1_i", pd.read_parquet(tmp_args.parquet_files[0], engine="fastparquet").columns)) # TODO clean up
    # TODO weird bug when we do --fcv1_i it thinks we did --fcv1_in

    for column in column_names:
        knn_parser.add_argument(f"--{column}", default=0, type=float, help=f"Specify the weight value for {column} (default: 0)")

    args = parser.parse_args()

    pretty_args = copy.deepcopy(args)
    delattr(pretty_args, "parquet_files")
    delattr(pretty_args, "model")

    if "knn" == args.model:
        weights = []
        for column in column_names:
            weights.append(getattr(args, column))
            delattr(args, column)

        args.weights = np.asarray(weights)

    return(args, pretty_args)



def read_parquets(parquet_files):
    # Read training data
    df_list = []

    for i, parquet_file in enumerate(parquet_files):
        df = pd.read_parquet(parquet_file, engine="fastparquet")
        df_list.append(df)

    return(df_list)



def process_dataframes(df_list, input_columns, predict_columns):
    # Separate x and y data
    x_df_list = []
    y_df_list = []

    for i, df in enumerate(df_list):
        x = df[input_columns].values
        y = df[predict_columns].values

        x_df_list.append(x)
        y_df_list.append(y)

    return(x_df_list, y_df_list)



def create_model(model_args):
    irrelevant_list = ["parquet_files", "model"] # TODO?
    model_type = model_args.model
    if model_type == "knn":
        relevant_args = {k: v for k, v in vars(model_args).items() if k not in irrelevant_list}
        return knn.KNNRegressor(**relevant_args)
    elif model_type == "xgb":
        relevant_args = {k: v for k, v in vars(model_args).items() if k not in irrelevant_list}
        return xgboost.XGBRegressor(**relevant_args)
    else:
        raise ValueError("invalid model type")



def shorten_parquet_files(parquet_files):
    shortened = []

    for filename in parquet_files:
        base_name = os.path.basename(filename)
        run_label = re.search(r"watch_data_(.*?)_clean", filename).group(1)
        shortened.append(run_label)

    return "_".join(shortened)



def save_data(filename, args, label_list, mse_list, mae_list, mape_list, n_list):
    # Check if file already contains headers
    file_exists = os.path.isfile(filename) and os.path.getsize(filename) > 0

    # Save the data to CSV
    with open(filename, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)

        # Write headers only if file is empty
        if not file_exists:
            headers = list(vars(args).keys()) + ["Label", "MSE", "MAE", "MAPE", "N"] * len(label_list)
            writer.writerow(headers)

        row = list(vars(args).values())

        for i in range(len(label_list)):
            row.extend([label_list[i], mse_list[i], mae_list[i], mape_list[i], n_list[i]])

        writer.writerow(row)



def generate_parameter_string(args):
    param_string = ""

    for arg, value in vars(args).items():
        arg_str = f"{arg}:{value};"

        if arg == "parquet_files":
            arg_str = f"{arg}:{shorten_parquet_files(value)};"

        param_string += arg_str

    return param_string



def has_matching_parameters(filename, pretty_values):
    if not os.path.isfile(filename):
        return False

    str_pretty_values = list(map(str, pretty_values))
    length = len(str_pretty_values)

    with open(filename, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if len(row) > 0 and row[:length] == str_pretty_values:
                return True
    return False



if __name__ == "__main__":
    save_dir = "./results/"
    predict_columns = ["fcv1_i"]

    args, pretty_args = cli_parser()

    csv_filename = save_dir + shorten_parquet_files(args.parquet_files) + "_" + args.model + ".csv"


    if not has_matching_parameters(csv_filename, list(vars(pretty_args).values())):
        df_list = read_parquets(args.parquet_files)

        # Get columns
        column_names = list(df_list[0].columns)
        input_columns = list(filter(lambda x: x not in predict_columns, column_names))

        x_df_list, y_df_list = process_dataframes(df_list, input_columns, predict_columns)

        # make lists to store all the data
        label_list = []
        mse_list = []
        mae_list = []
        mape_list = []
        n_list = []

        # K fold on each of the dataframes
        for i, _ in enumerate(df_list):
            # Get testing data
            x_test = x_df_list[i]
            y_test = y_df_list[i]

            # Get training data
            x_train = np.concatenate(x_df_list[:i] + x_df_list[i + 1:])
            y_train = np.concatenate(y_df_list[:i] + y_df_list[i + 1:])

            # Create and train model
            model = create_model(args)
            model.fit(x_train, y_train)

            # Predict on the test data with a progress bar
            y_pred = np.zeros_like(y_test)
            y_len = len(y_pred)
            for j, x in tqdm(enumerate(x_test), total=y_len, desc="Predicting"):
                new_x = x.reshape(1, -1)
                y_pred[j] = model.predict(new_x)

            # Calculate mean losses
            mse_loss = np.mean((y_pred - y_test) ** 2)
            mae_loss = np.mean(np.abs(y_pred - y_test))
            mape_loss = np.mean(np.abs((y_pred - y_test) / y_test)) * 100

            label_list.append(os.path.basename(args.parquet_files[i]))
            mse_list.append(mse_loss)
            mae_list.append(mae_loss)
            mape_list.append(mape_loss)
            n_list.append(y_len)


        def weighted_average(val_list, count_list):
            total = 0
            total_count = 0

            for i, _ in enumerate(val_list):
                total += val_list[i] * count_list[i]
                total_count += count_list[i]

            return(total/total_count)


        label_list = ["agg"] + label_list
        mse_list = [weighted_average(mse_list, n_list)] + mse_list
        mae_list = [weighted_average(mae_list, n_list)] + mae_list
        mape_list = [weighted_average(mape_list, n_list)] + mape_list
        n_list = [sum(n_list)] + n_list


        save_data(csv_filename, pretty_args, label_list, mse_list, mae_list, mape_list, n_list)

    else:
        print("skipping")
