from typing import Union, Sequence, Callable, Tuple, Optional, List
import pandas as pd


def read_file(file_path: str):
    """
    Reads .csv or .parquet file into a pandas dataframe
    Args:
        file_path (str): path to file

    Returns:
        A pandas dataframe containing file contents
    Raises:
            ValueError: If file extension is not supported
            FileNotFoundError: If file is not found at the given path.
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Invalid file type: {file_path}. File must be a .csv or .parquet file")


def run_select(df: pd.DataFrame, run_selection: Union[float, Sequence[float]]) -> pd.DataFrame:
    """
    Select the desired runs for the dataset
    Args:
        df (pd.DataFrame): Full dataset
        run_selection (Union[float, Sequence[float]]): desired runs

    Returns:
        A new dataframe with only the selected runs

    Raises:
        ValueError: if an invalid form of run selection is inputted
    """

    if type(run_selection) in [int, float]:
        comparator = lambda col, value: df == value
    elif type(run_selection) is List[float]:
        comparator = lambda col, values: df.isin(values)
    else:
        raise ValueError(f"Invalid run selection: {run_selection}. Must be float or List[float]")
    return select_rows(df, "run_id", run_selection, comparator)


def select_rows(df: pd.DataFrame, column_name: str, value: Union[float, Sequence[float]],
                comparator: Callable[[pd.DataFrame, Union[float, Sequence[float]]], bool]) -> pd.DataFrame:
    """
    Selects columns from a Dataframe based on a value and a given comparator function. It is important that
    the comparator function works with the type of value used (i.e. a single value or sequence of values)
    Args:
        df: The dataframe to select from
        column_name (str): The name of the column to use with comparator
        value (Union[float, Sequence[float]]): The value to compare with
        comparator (Callable[[pd.DataFrame, int], bool]): A function that takes in a dataframe and a value,
         and returns a boolean mask

    Returns:
        new_df (pd.DataFrame): A new dataframe consisting of the selected rows
    """
    mask = comparator(df[column_name], value)
    new_df = df[mask]
    return new_df


def standardize(df: pd.DataFrame, mean: Optional[pd.Series] = None,
                std: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Tuple[pd.Series, pd.Series]]:
    """
    Standardize a dataframe's columns, and return the parameters
    Args:
        df (pd.DataFrame): pandas DataFrame to modify
        mean (Optional[pd.Series]): optional provided mean
        std (Optional[pd.Series]): optional provided std

    Returns:
        A tuple containing the standardized dataframe as well as the parameters used to scale it. In the following form:
        (scaled_df, (mean, std))
    """
    assert type(mean) == type(std), "mean and std should both be pd.Series or both be None"
    if mean is None:
        mean, std = df.min(), df.max()
    scaled_df = (df - mean) / std
    return scaled_df, (mean, std)


def min_max_scale(df: pd.DataFrame, min_val: Optional[pd.Series] = None,
                  max_val: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Tuple[pd.Series, pd.Series]]:
    """
    Min-Max scale a dataframe's columns, and return the parameters
    Args:
        max_val (Optional[pd.Series]): optional provided max
        min_val (Optional[pd.Series]): optional provided min
        df (pd.DataFrame): pandas DataFrame to modify


    Returns:
        A tuple containing the scaled dataframe as well as the parameters used to scale it. In the following form:
        (scaled_df, (min_val, max_val))
    """
    assert type(min_val) is type(max_val), "min_val and max_val should both be pd.Series or both be None"
    if min_val is None:
        min_val, max_val = df.min(), df.max()
    df = (df - min_val) / (max_val - min_val)
    return df, (min_val, max_val)


def create_differential_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new differential features in dataframe
    Args:
        df (pd.DataFrame): dataframe to modify

    Returns:
        A modified dataframe with differential features
    """
    for col in df.columns:
        df[f'{col}_diff'] = df[col].diff()
    return df


def make_differential(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert to differential dataset
    Args:
        df (pd.DataFrame): dataframe to convert

    Returns:
        A modified dataframe with differential features
    """
    df = df.diff()
    return df


def generate_smoother(window_size: int) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Creates a smoothing function for a defined window_size
    Args:
        window_size: size of smoothing window

    Returns:
        A new function that smooths a dataframe over a specified window
    """
    def smooth_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        df = df.rolling(window=window_size).mean()
        return df
    return smooth_dataframe


def generate_lag_fn(lag_steps: int) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Creates a function to add lag features
    Args:
        lag_steps: number of lag steps

    Returns:
        A new function that creates lag features up to a specified number of lag features
    """
    def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
        for lag in range(1, lag_steps + 1):
            for col in df:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        return df
    return create_lag_features


def generate_rolling_stats_fn(window_size: int) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Creates a function to add rolling statistics features
    Args:
        window_size: window size for rolling statistics

    Returns:
        A new function that creates rolling statistic features for a specified window

    """
    def create_rollings_statistics_features(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size).mean()
            df[f'{col}_rolling_std'] = df[col].rolling(window=window_size).std()
        return df
    return create_rollings_statistics_features
