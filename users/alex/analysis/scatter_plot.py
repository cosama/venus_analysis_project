import argparse
import os
import re

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd



def check_files_with_prefix(directory, prefix):
    for filename in os.listdir(directory):
        if filename.startswith(prefix):
            return True
    return False



def save_rotated_scatter_plot(df_list, filename_prefix, x_col, y_col, z_col, a_col, step_size, elevations, dimension_limits):
    # Combine the parquet files
    df_combined = pd.concat(df_list)

    # Extract the column values
    x = df_combined[x_col].values
    y = df_combined[y_col].values
    z = df_combined[z_col].values
    a = df_combined[a_col].values

    # Create the 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=a, cmap="cool")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)

    # Set the minimum and maximum values for each dimension
    if x_col in dimension_limits:
        ax.set_xlim(dimension_limits[x_col]["min"], dimension_limits[x_col]["max"])
    if y_col in dimension_limits:
        ax.set_ylim(dimension_limits[y_col]["min"], dimension_limits[y_col]["max"])
    if z_col in dimension_limits:
        ax.set_zlim(dimension_limits[z_col]["min"], dimension_limits[z_col]["max"])

    # Save pictures for each rotation angle
    for angle in range(0, 360, int(step_size)):
        for elevation in elevations:
            ax.view_init(elev=elevation, azim=angle)
            filename = f"{filename_prefix}_elevation_{elevation}_angle_{angle}.png"
            plt.savefig(filename)
            print(f"Saved {filename}")

    plt.close(fig)



if __name__ == "__main__":
    # data directory
    plot_dir = "./plots/"

    # define the dimension limits dictionary
    dimension_limits = {
        "bias_v": {"min": 0, "max": 80},
        "bias_i": {"min": 0, "max": 20},
        "inj_mbar": {"min": 0, "max": 3e-7},
        "extraction_i": {"min": 0, "max": 12},
    }

    # parse cli
    parser = argparse.ArgumentParser(description="Generate and save rotated scatter plots")
    parser.add_argument("--x_col", type=str, required=True, help="Name of the column for 'x'")
    parser.add_argument("--y_col", type=str, required=True, help="Name of the column for 'y'")
    parser.add_argument("--z_col", type=str, required=True, help="Name of the column for 'z'")
    parser.add_argument("--a_col", type=str, required=True, help="Name of the column for 'a'")
    parser.add_argument("--step_size", default=10, type=int, help="Size of rotation step in degrees")
    parser.add_argument("--elevations", default=[30], type=int, nargs="+", help="Elevations of the camera")
    parser.add_argument("--filenames", nargs="+", required=True, help="List of Parquet filenames")
    args = parser.parse_args()

    # Use regex to extract the run from the filenames
    shortened_parquet_filenames = [re.search(r"watch_data_(.*?)_clean", os.path.basename(parquet_file)).group(1) for parquet_file in args.filenames]
    parquet_filenames_str = "_".join(shortened_parquet_filenames)

    filename_prefix = f"scatter_plot_{args.x_col}_{args.y_col}_{args.z_col}_{args.a_col}_{parquet_filenames_str}_"

    if not check_files_with_prefix(plot_dir, filename_prefix):
        # Read the parquet files into memory
        df_list = [pd.read_parquet(parquet_file) for parquet_file in args.filenames]

        # run function
        save_rotated_scatter_plot(
            df_list,
            plot_dir + filename_prefix,
            args.x_col,
            args.y_col,
            args.z_col,
            args.a_col,
            args.step_size,
            args.elevations,
            dimension_limits
        )
