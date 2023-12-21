from typing import List, Optional
import torch
from torch.utils.data import Dataset
import pandas as pd


class VenusDataset(Dataset):
    """
    Custom dataset that can read from .csv or .parquet files and
    apply data preprocessing, smoothing, and differential conversion.

    Attributes:
        df : pandas dataframe containing dataset
        inputs : list of input features for dataset
        outputs : list of output features for dataset
    """

    def __init__(self,
                 file_path: str,
                 input_columns: List[str],
                 output_columns: List[str],
                 run_id: Optional[float] = None,
                 preprocess_data: str = None,
                 smoothing_window: Optional[int] = None,
                 make_differential: bool = False) -> None:
        """
        Initialize the Dataset.

        Args:
            file_path (str): Path to the .csv or .parquet file.
            input_columns (List[str]): List of column names to be used as input features.
            output_columns (List[str]): List of column names to be used as output features.
            run_id (Optional[int]): Int to select only data from "run_id"
            preprocess_data (str): Preprocess dataset by "standardize" or "min-max", None to disable
            smoothing_window (Optional[int]): Window size for smoothing, None to disable
            make_differential (bool): If True, convert data to differential format

        Raises:
            ValueError: If file extension is not supported or invalid preprocess type
            FileNotFoundError: If file is not found at the given path.
        """

        if file_path.endswith('.csv'):
            self.df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            self.df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Invalid file type: {file_path}. File must be a .csv or .parquet file")

        if run_id is not None:
            self.df = self.df[self.df["run_id"] == run_id]
        if preprocess_data is not None:
            if preprocess_data == "standardize":
                self.df = (self.df - self.df.mean()) / self.df.std()
            elif preprocess_data == "min-max":
                self.df = (self.df - self.df.min())/(self.df.max() - self.df.min())
            else:
                raise ValueError(f"Invalid preprocess: {preprocess_data}. Must be \'standardize\' or \'min-max\'")
        if smoothing_window is not None:
            self.df = self.df.rolling(window=smoothing_window).mean().dropna()
        if make_differential:
            self.df = self.df.diff().dropna()

        self.inputs = self.df[input_columns]
        self.outputs = self.df[output_columns]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = self.inputs.iloc[idx]
        outputs = self.outputs.iloc[idx]

        inputs = torch.tensor(inputs.values, dtype=torch.float32)
        outputs = torch.tensor(outputs.values, dtype=torch.float32)

        return inputs, outputs


class VenusTimeSeriesDataset(Dataset):
    """
    Custom dataset that can read from .csv or .parquet files and
    apply data preprocessing, smoothing, and differential conversion.
    Includes time-series specific feature engineering like rolling window statistics
    and lag features.

    Attributes:
        df : pandas dataframe containing dataset
        inputs : list of input features for dataset
        outputs : list of output features for dataset
    """
    def __init__(self,
                 file_path: str,
                 input_columns: List[str],
                 output_columns: List[str],
                 run_id: Optional[float] = None,
                 preprocess_data: str = None,
                 smoothing_window: Optional[int] = None,
                 make_differential: bool = False,
                 lag_steps: int = None,
                 rolling_window_stats: int = None
                 ) -> None:

        """
        Initialize the Dataset.

        Args:
            file_path (str): Path to the .csv or .parquet file.
            input_columns (List[str]): List of column names to be used as input features.
            output_columns (List[str]): List of column names to be used as output features.
            run_id (Optional[int]): Int to select only data from "run_id"
            preprocess_data (str): Preprocess dataset by "standardize" or "min-max", None to disable
            smoothing_window (Optional[int]): Window size for smoothing, None to disable
            make_differential (bool): If True, convert data to differential format
            lag_steps (int): Number of lag step features to include, None to disable
            rolling_window_stats: Window size for rolling statistics to be calculated over, None to disable

        Raises:
            ValueError: If file extension is not supported or invalid preprocess type
            FileNotFoundError: If file is not found at the given path.
        """

        if file_path.endswith('.csv'):
            self.df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            self.df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Invalid file type: {file_path}. File must be a .csv or .parquet file")
        if run_id is not None:
            self.df = self.df[self.df["run_id"] == run_id]
        if preprocess_data is not None:
            if preprocess_data == "standardize":
                self.df = (self.df - self.df.mean()) / self.df.std()
            elif preprocess_data == "min-max":
                self.df = (self.df - self.df.min()) / (self.df.max() - self.df.min())
            else:
                raise ValueError(f"Invalid preprocess: {preprocess_data}. Must be \'standardize\' or \'min-max\'")

        if smoothing_window is not None:
            self.df = self.df.rolling(window=smoothing_window).mean()
        if make_differential:
            self.df = self.df.diff()

        if lag_steps:
            for lag in range(1, lag_steps + 1):
                for col in input_columns:
                    self.df[f"{col}_lag_{lag}"] = self.df[col].shift(lag)

        if rolling_window_stats:
            for col in input_columns:
                self.df[f'{col}_rolling_mean'] = self.df[col].rolling(window=rolling_window_stats).mean()
                self.df[f'{col}_rolling_std'] = self.df[col].rolling(window=rolling_window_stats).std()

        self.inputs = self.df[[col for col in self.df.columns if col.startswith(tuple(input_columns))]].fillna(0)
        self.outputs = self.df[output_columns].fillna(0)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = self.inputs.iloc[idx]
        outputs = self.outputs.iloc[idx]

        inputs = torch.tensor(inputs.values, dtype=torch.float32)
        outputs = torch.tensor(outputs.values, dtype=torch.float32)

        return inputs, outputs
