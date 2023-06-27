import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

if __name__ == "__main__":
    data_dir = "../../../clean_data/vary_pressure_and_bias_voltage/"

    parser = argparse.ArgumentParser(description="Random Forest Regression")
    parser.add_argument("--train_csv_files", required=True, nargs="+", help="Training CSV files")
    parser.add_argument("--column_names_file", default="column_names", help="Column names CSV file")
    parser.add_argument("--predict_columns", default=["fcv1_i"], nargs="+", help="Columns to predict")
    args = parser.parse_args()

    train_csv_files = list(map(lambda x: data_dir + x, args.train_csv_files))
    column_names_file = data_dir + args.column_names_file
    predict_columns = args.predict_columns

    # get other columns
    column_names = list(pd.read_csv(column_names_file, delimiter=", "))
    input_columns = list(filter(lambda x: x not in predict_columns, column_names))

    # Read training data
    tmp = []
    for train_csv_file in train_csv_files:
        tmp.append(pd.read_csv(train_csv_file, names=column_names, delimiter=" "))
    train_df = pd.concat(tmp)
    x_train = train_df[input_columns].values
    y_train = train_df[predict_columns].values

    # Create and train the Random Forest regression model
    rf_model = RandomForestRegressor(n_estimators=10, n_jobs=-1)
    rf_model.fit(x_train, y_train)

    # Get feature importances
    feature_importances = rf_model.feature_importances_

    # Print feature importance values
    print("Feature Importance:")
    for feature_name, importance in zip(input_columns, feature_importances):
        print(f"{feature_name}: {importance}")
