import os
import argparse

import pandas as pd

clean_data_dir = "../../clean_data/vary_pressure_and_bias_voltage/"
clean_suffix = "_clean"
separator = " "

# Create the argument parser
parser = argparse.ArgumentParser(description='Cleans data files.')
parser.add_argument('file_paths', metavar='FILE', nargs='+',
                    help='file paths to clean')

# Parse the command-line arguments
args = parser.parse_args()

# Process each file path
for file_path in args.file_paths:

    clean_file_path = clean_data_dir + file_path + clean_suffix

    with open(file_path, "r") as read_file:
        with open(clean_file_path, "w") as write_file:
            first_line = read_file.readline().strip()
            num_separators = first_line.count(separator)
            write_file.write(first_line + "\n")  # Write the first line

            for line in read_file.readlines():
                fixed_line = line.strip()
                if fixed_line.count(separator) == num_separators: # TODO
                    write_file.write(fixed_line + "\n")

    # because for the first file we collected the times incorrectly
    if file_path == "watch_data":
        timeinfo = pd.read_csv("timeinfo")
        data = pd.read_csv(clean_data_dir + "watch_data" + clean_suffix, header=None, index_col=False, delimiter=" ")

        data[1] = timeinfo

        data.to_csv(clean_data_dir + "watch_data_00_clean", header=False, index=False, sep=" ")

        os.remove(clean_data_dir + "watch_data_clean")
