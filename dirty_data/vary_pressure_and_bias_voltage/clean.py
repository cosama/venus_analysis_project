import os
import argparse

import pandas as pd

clean_data_dir = "../../clean_data/vary_pressure_and_bias_voltage/"
clean_suffix = "_clean"
tmp_suffix = "_tmp"
separator = " "
column_names_file = "column_names"

# Create the argument parser
parser = argparse.ArgumentParser(description='Cleans data files.')
parser.add_argument('--file_paths', required=True, nargs='+', help='file paths to clean')

# Parse the command-line arguments
args = parser.parse_args()

# Parse column names file
column_names = list(pd.read_csv(column_names_file, delimiter=", ", engine="python"))

# Get rid of all the spaces
for file_path in args.file_paths:

    tmp_file_path = file_path + tmp_suffix

    with open(file_path, "r") as read_file:
        with open(tmp_file_path, "w") as write_file:
            first_line = read_file.readline().strip()
            num_separators = first_line.count(separator)
            write_file.write(first_line + "\n")  # Write the first line

            for line in read_file.readlines():
                fixed_line = line.strip()
                if fixed_line.count(separator) == num_separators: # TODO
                    write_file.write(fixed_line + "\n")

# Turn into parquet files
for file_path in args.file_paths:
    tmp_file_path = file_path + tmp_suffix

    data = pd.read_csv(tmp_file_path, header=None, names=column_names, index_col=False, delimiter=" ")

    clean_file_path = clean_data_dir + file_path + clean_suffix

    # because for the first file we collected the times incorrectly
    if file_path == "watch_data":
        clean_file_path = clean_data_dir + "watch_data_00" + clean_suffix
        timeinfo = pd.read_csv("timeinfo")
        data["time"] = timeinfo

    data.to_parquet(clean_file_path, compression=None)
    os.remove(tmp_file_path)
