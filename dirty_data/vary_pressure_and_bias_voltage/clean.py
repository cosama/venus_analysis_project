import os
import sys

import pandas as pd

clean_data_dir = "../../clean_data/vary_pressure_and_bias_voltage/"

clean_suffix = "_clean"

# Get the file paths from command-line arguments
file_paths = sys.argv[1:]

# strip beginning and end of each line
for file_path in file_paths:
    clean_file_path = clean_data_dir + file_path + clean_suffix
    with open(file_path, "r") as read_file:
        with open(clean_file_path, "w") as write_file:
            for line in read_file.readlines():
                write_file.write(line.strip() + "\n")

    if file_path == "watch_data":
        timeinfo = pd.read_csv("timeinfo")
        data = pd.read_csv(clean_data_dir + "watch_data" + clean_suffix, header=None, index_col=False, delimiter=" ")

        data[1] = timeinfo

        data.to_csv(clean_data_dir + "watch_data_00_clean", header=False, index=False, sep=" ")

        os.remove(clean_data_dir + "watch_data_clean")

