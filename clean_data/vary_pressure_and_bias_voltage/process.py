import os
import argparse

import pandas as pd

put_data_dir = "../../processed_data/vary_pressure_and_bias_voltage/"
new_suffix = "_processed"
old_suffix = "_clean"
minimum_output = 1e-3

# Create the argument parser
parser = argparse.ArgumentParser(description="Processes data files.")
parser.add_argument("--file_paths", required=True, nargs="+", help="file paths to process")

# Parse the command-line arguments
args = parser.parse_args()

# Get rid of all the spaces
for file_path in args.file_paths:

    put_file_path = put_data_dir + os.path.basename(file_path[:-1 * len(old_suffix)]) + new_suffix

    data = pd.read_parquet(file_path)

    maximum = data["fcv1_i"].max()
    minimum = data["fcv1_i"].min()

    data["fcv1_i_normalized"] = (data["fcv1_i"] - minimum) / (maximum - minimum) + minimum_output

    data.to_parquet(put_file_path, compression=None)

