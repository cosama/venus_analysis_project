import argparse
import os
import re

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from PIL import Image


def filename_generator(x_col, y_col, z_col, a_col, parquet_filenames_str, elevations, step_size):
    for elevation in elevations:
        for angle in range(0, 360, int(step_size)):
            yield(f"scatter_plot_x_{x_col}_y_{y_col}_z_{z_col}_a_{a_col}_files_{parquet_filenames_str}_elevation_{elevation}.gif", angle, elevation)



def check_file_exists_in_dir(directory, check_filename):
    for filename in os.listdir(directory):
        if filename == check_filename:
            return True
    return False



def save_rotated_scatter_plot(df_list, plot_dir, parquet_filenames_str, x_col, y_col, z_col, a_col, step_size, elevations, dimension_limits):
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

    # Ideally would like to make the 1e-7 for inj_mbar appear in a better location
    # https://github.com/matplotlib/matplotlib/issues/4476

    # Set the minimum and maximum values for each dimension
    if x_col in dimension_limits:
        ax.set_xlim(dimension_limits[x_col]["min"], dimension_limits[x_col]["max"])
    if y_col in dimension_limits:
        ax.set_ylim(dimension_limits[y_col]["min"], dimension_limits[y_col]["max"])
    if z_col in dimension_limits:
        ax.set_zlim(dimension_limits[z_col]["min"], dimension_limits[z_col]["max"])

    images = {}  # Store the images for each elevation

    # Save pictures for each rotation angle
    for (filename, angle, elevation) in filename_generator(x_col, y_col, z_col, a_col, parquet_filenames_str, elevations, step_size):
        ax.view_init(elev=elevation, azim=angle)
        fig.canvas.draw()

        # Convert the plot to an image
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.setdefault(elevation, []).append(Image.fromarray(image))

        print(f"Generated angle={angle} elevation={elevation}")

    # Save the images as GIF files for each elevation
    for (filename, angle, elevation) in filename_generator(x_col, y_col, z_col, a_col, parquet_filenames_str, elevations, step_size=360):
        images[elevation][0].save(plot_dir + filename, save_all=True, append_images=images[elevation][1:], optimize=False, duration=100, loop=0)

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

    # Check if we have all the files we want to generate
    have_all_files = True
    for filename, _, _ in filename_generator(args.x_col, args.y_col, args.z_col, args.a_col, parquet_filenames_str, args.elevations, args.step_size): 
        have_all_files = have_all_files and check_file_exists_in_dir(plot_dir, filename)

    # Create the files if we don't have them
    if not have_all_files:
        # Read the parquet files into memory
        df_list = [pd.read_parquet(parquet_file) for parquet_file in args.filenames]

        # run function
        save_rotated_scatter_plot(
            df_list,
            plot_dir,
            parquet_filenames_str,
            args.x_col,
            args.y_col,
            args.z_col,
            args.a_col,
            args.step_size,
            args.elevations,
            dimension_limits
        )
