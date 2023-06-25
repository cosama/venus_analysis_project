import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import re



def save_rotated_scatter_plot(csv_files, column_names_file, w_col, x_col, y_col, z_col, step_size, dimension_limits):
    # Read the CSV files and column names
    names = list(pd.read_csv(data_dir + column_names_file, delimiter=", "))
    dfs = [pd.read_csv(data_dir + csv_file, names=names, delimiter=" ") for csv_file in csv_files]
    df_combined = pd.concat(dfs)

    # Use regex to extract the run number from the filenames
    shortened_csv_filenames = [re.search(r"watch_data_(.*?)_clean", csv_file).group(1) for csv_file in csv_files]

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
    for angle in range(0, 360, int(step_size)):
        ax.view_init(elev=30, azim=angle)  # Set the elevation and azimuth angles
        csv_filenames_str = "_".join(shortened_csv_filenames)
        filename = f"{plot_dir}scatter_plot_{w_col}_{x_col}_{y_col}_{z_col}_{csv_filenames_str}_angle_{angle}.png"
        plt.savefig(filename)
        print(f"Saved {filename}")

    plt.close(fig)



if __name__ == "__main__":
    # data directory
    data_dir = "../../../clean_data/vary_pressure_and_bias_voltage/"
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
    parser.add_argument("--w_col", type=str, required=True, help="Name of the column for 'w'")
    parser.add_argument("--x_col", type=str, required=True, help="Name of the column for 'x'")
    parser.add_argument("--y_col", type=str, required=True, help="Name of the column for 'y'")
    parser.add_argument("--z_col", type=str, required=True, help="Name of the column for 'z'")
    parser.add_argument("--step_size", default=10, type=int, help="Size of rotation step in degrees")
    parser.add_argument("--column_names_file", default="column_names", type=str, help="CSV file that has the column names")
    parser.add_argument("--filenames", nargs="+", required=True, help="List of CSV filenames")
    args = parser.parse_args()

    # run function
    save_rotated_scatter_plot(
        args.filenames,
        args.column_names_file,
        args.w_col,
        args.x_col,
        args.y_col,
        args.z_col,
        args.step_size,
        dimension_limits
    )
