import argparse
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data_dir = "../../../clean_data/vary_pressure_and_bias_voltage/"
plot_dir = "./plots/"

# Define the dimension limits dictionary
dimension_limits = {
    "bias_v": {"min": 0, "max": 175},
    "bias_i": {"min": 0, "max": 20},
    "inj_mbar": {"min": 0, "max": 2e-6},
    "extraction_i": {"min": 0, "max": 12},
}


def save_rotated_scatter_plot(csv_files, column_names_file, w_col, x_col, y_col, z_col, stepsize, csv_filenames):
    # Read the CSV files and column names
    names = list(pd.read_csv(data_dir + column_names_file, delimiter=", "))
    dfs = [pd.read_csv(data_dir + csv_file, names=names, delimiter=" ") for csv_file in csv_files]
    df_combined = pd.concat(dfs)

    # if not we reach the limit on file lengths in ext4 filesystems
    # just grabs the run number
    # could use regex, but this is currently a quick and dirty script
    shortened_csv_filenames = list(map(lambda x: x[11:13], csv_filenames))

    # Extract the column values
    w = df_combined[w_col].values
    x = df_combined[x_col].values
    y = df_combined[y_col].values
    z = df_combined[z_col].values

    # Create the 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(w, x, y, c=z, cmap="hot")
    ax.set_xlabel(w_col)
    ax.set_ylabel(x_col)
    ax.set_zlabel(y_col)

    # Set the minimum and maximum values for each dimension
    if w_col in dimension_limits:
        ax.set_xlim(dimension_limits[w_col]["min"], dimension_limits[w_col]["max"])
    if x_col in dimension_limits:
        ax.set_ylim(dimension_limits[x_col]["min"], dimension_limits[x_col]["max"])
    if y_col in dimension_limits:
        ax.set_zlim(dimension_limits[y_col]["min"], dimension_limits[y_col]["max"])

    # Save pictures for each rotation angle
    for angle in range(0, 360, int(stepsize)):
        ax.view_init(elev=30, azim=angle)  # Set the elevation and azimuth angles
        csv_filenames_str = "_".join(shortened_csv_filenames)
        filename = f"{plot_dir}scatter_plot_{w_col}_{x_col}_{y_col}_{z_col}_{csv_filenames_str}_angle_{angle}.png"
        plt.savefig(filename)
        print(f"Saved {filename}")

    plt.close(fig)

def parse_arguments_file(args_file, delimiter=","):
    arguments_list = []
    with open(args_file, "r") as file:
        lines = file.readlines()
    for line in lines:
        arguments = line.strip().split(delimiter)
        arguments_list.append(arguments)
    return arguments_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save rotated scatter plots")
    parser.add_argument("args_file", type=str, help="File containing arguments")

    args = parser.parse_args()

    argument_sets = parse_arguments_file(args.args_file, ", ")
    for arguments in argument_sets:
        column_names_file, w_col, x_col, y_col, z_col, stepsize, *csv_files = arguments

        save_rotated_scatter_plot(csv_files, column_names_file, w_col, x_col, y_col, z_col, stepsize, csv_files)

