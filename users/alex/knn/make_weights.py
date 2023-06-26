import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm



def calculate_pairwise_gradients(data, predictive_variable):
    num_cols = data.shape[1]
    gradients = np.zeros((num_cols, num_cols))

    predictive_column = data[predictive_variable].values

    progress_bar = tqdm(total=(num_cols * (num_cols - 1) // 2), desc="Calculating pairwise gradients")

    for i in range(num_cols):
        for j in range(i + 1, num_cols):
            feature_column_i = data.iloc[:, i].values
            feature_column_j = data.iloc[:, j].values

            gradient = np.mean(np.abs((predictive_column[i] - predictive_column[j]) / (feature_column_i - feature_column_j)))
            gradient = np.nan_to_num(gradient)
            gradients[i, j] = gradient
            gradients[j, i] = gradient

            progress_bar.update(1)

    progress_bar.close()
    return pd.DataFrame(gradients, index=data.columns, columns=data.columns)



def main():
    data_dir = "../../../clean_data/vary_pressure_and_bias_voltage/"

    parser = argparse.ArgumentParser(description="Calculate pairwise gradients for feature weights.")
    parser.add_argument("--csv_files", nargs="+", required=True, help="Input CSV file(s)")
    parser.add_argument("--column_names_file", default="column_names", help="Column names CSV file")
    parser.add_argument("--predictive_variable", default="fcv1_i", help="Variable for which to calculate gradients")
    args = parser.parse_args()

    column_names = list(pd.read_csv(data_dir + args.column_names_file, delimiter=", "))

    # Read CSV files
    data_list = []

    for csv_file in args.csv_files:
        data_list.append(pd.read_csv(data_dir + csv_file, names=column_names, delimiter=" "))

    all_data = pd.concat(data_list)

    # Calculate pairwise gradients
    gradients = calculate_pairwise_gradients(all_data, args.predictive_variable)

    print("Pairwise Gradients:")
    print(gradients.loc[args.predictive_variable].values)



if __name__ == "__main__":
    main()
