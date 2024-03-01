from typing import List, Optional, Union, Sequence, Tuple, Callable
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


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


def run_select(df: pd.DataFrame, run_selection: Union[int, float, Sequence[Union[int, float]]]) -> pd.DataFrame:
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
        mask = df["run_id"] == run_selection
    elif type(run_selection) in [list, tuple]:
        mask = df["run_id"].isin(run_selection)
    else:
        raise ValueError(f"Invalid run selection: {run_selection}. Must be int, float, Sequence[Union[float, int]]")
    return df[mask]


class VenusDataset(Dataset):
    """
     Custom dataset that can read from .csv or .parquet files. Includes functionality to pass in a scaler and
     transforms to preprocess dataset. Stores scaling parameters in order to apply them to test/validation datasets

     If the sequence length is set to anything but the default 0, then this functions as a time-series style dataset, where
     a sequence of inputs leading up to a target output is returned from __getitem__ instead of all inputs and outputs
     at a single timestep. The __len__ method is also modified so there are no invalid accesses.

     Attributes:
        df : pandas dataframe containing dataset
        inputs : dataframe of input features
        outputs : dataframe of output features
        scale_function : function to scale data
        transforms : sequence of transforms applied to data
        sequence_length : the length of the sequence returned by __getitem__
    """

    def __init__(self,
                 file_path: str,
                 input_columns: List[str],
                 output_columns: List[str],
                 run_selection: Union[int, float, Sequence[float]] = None,
                 scaler: Callable[[pd.DataFrame, Optional[pd.Series], Optional[pd.Series]], Tuple[pd.DataFrame, Tuple[pd.Series, pd.Series]]] = None,
                 transforms: Sequence[Callable[[pd.DataFrame], pd.DataFrame]] = None,
                 sequence_length: int = 0
                 ):
        """
        Initializes a dataset with specified run and input/output columns then scales and transforms it

        Args:
            file_path (str): path to .csv or .parquet file containing dataset
            input_columns (List[str]): list of input column names
            output_columns (List[str]): list of output column names
            run_selection (Union[int, float, Sequence[float]]): run or list of runs to use
            scaler (Callable[[pd.DataFrame, Optional[pd.Series], Optional[pd.Series]],
                                  Tuple[pd.DataFrame, Tuple[pd.Series, pd.Series]]]): function to scale data
            transforms (Sequence[Callable[[pd.DataFrame], pd.DataFrame]]): list of transform functions
            sequence_length (int): length of sequence to return
        """
        # TODO fix issue where standardizing scales run_id columns so cant be referenced
        self.df = read_file(file_path)
        if run_selection:
            self.df = run_select(self.df, run_selection)
        if scaler:
            self.scale_function = scaler
            self.df, self.scale_params = scaler(self.df, None, None)
        if transforms:
            self.transforms = transforms
            for transform in transforms:
                self.df = transform(self.df)

        self.sequence_length = sequence_length
        self.inputs = self.df[[col for col in self.df.columns if col.startswith(tuple(input_columns))]].fillna(0)
        self.outputs = self.df[output_columns].fillna(0)

    def __len__(self) -> int:
        return len(self.df) - self.sequence_length

    def dataset_size(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = self.inputs.iloc[idx:idx + self.sequence_length] if self.sequence_length else self.inputs.iloc[idx]
        outputs = self.outputs.iloc[idx + self.sequence_length] if self.sequence_length else self.outputs.iloc[idx]

        inputs = torch.tensor(inputs.values, dtype=torch.float32)
        outputs = torch.tensor(outputs.values, dtype=torch.float32)

        return inputs, outputs

    def apply_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply data scaling to another dataset using this dataset's parameters
        Args:
            df: test/validation dataset

        Returns:
            A scaled dataframe
        """
        param1, param2 = self.scale_params
        return self.scale_function(df, param1, param2)[0]

    def apply_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply data transforms to another dataframe, intended for use on test/validation datasets
        Args:
            df: test/validation dataset

        Returns:
            A transformed dataframe
        """
        for transform in self.transforms:
            df = transform(df)
        return df

    def to_numpy(self):
        """
        Get numpy arrays containing the data for use with sci-kit learn models using .fit(X_data, y_data)
        Returns:
            A tuple containing numpy arrays for the inputs and outputs

        """
        return self.inputs.values, self.outputs.values

    def to_tensor(self):
        """
        Get torch tensors containing the data
        Returns:
            A tuple containing numpy arrays for the inputs and outputs

        """
        inputs, outputs = self.to_numpy()
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(outputs, dtype=torch.float32)

    def get_runs(self, run_ids: Sequence[Union[float, int]]) -> dict:
        """
        Retrieves a dictionary of run_ids paired with corresponding data as numpy arrays
        Args:
            run_ids (Sequence[Union[float, int]]): Which runs to select

        Returns:
            A dictionary containing run_ids paired with corresponding data
        """
        runs_data = {}
        for run_id in run_ids:
            run_df = self.df[self.df["run_id"] == run_id]
            input_columns = [col for col in run_df.columns if col in self.inputs.columns]
            output_columns = [col for col in run_df.columns if col in self.outputs.columns]
            runs_data[run_id] = (run_df[input_columns].to_numpy(), run_df[output_columns].to_numpy())
        return runs_data

    def get_run_splits(self, run_ids: Sequence[Union[float]], validation_size: Optional[float] = 0.2,
                       random_state: Optional[int] = None, shuffle=True) -> Tuple:
        """
        Retrieves runs data and splits into training and validation sets with the split occurring in each run
        individually
        Args:
            run_ids (Sequence): Which runs to select
            validation_size (float): Which percentage of data to make validation
            random_state (int): Optional parameter to set random state, defaults to None
            shuffle (bool): Whether to shuffle the run before splitting, defaults to True

        Returns:
            Tuple of numpy arrays containing
            (train_input, train_output, validation_input, validation_output)
        """
        X_train_list, X_val_list, y_train_list, y_val_list = [], [], [], []
        runs_data = self.get_runs(run_ids)
        for run_id, (inputs, outputs) in runs_data.items():
            X_train, X_val, y_train, y_val = train_test_split(
                inputs, outputs, test_size=validation_size, random_state=random_state, shuffle=shuffle
            )
            X_train_list.append(X_train)
            X_val_list.append(X_val)
            y_train_list.append(y_train)
            y_val_list.append(y_val)

        train_inputs = np.concatenate(X_train_list, axis=0)
        validation_inputs = np.concatenate(X_val_list, axis=0)
        train_outputs = np.concatenate(y_train_list, axis=0)
        validation_outputs = np.concatenate(y_val_list, axis=0)

        return train_inputs, train_outputs, validation_inputs, validation_outputs



