from typing import List, Optional, Union, Sequence, Tuple, Callable
import torch
from torch.utils.data import Dataset
import pandas as pd
from preprocessing import run_select, read_file


class VenusDataset(Dataset):
    """
     Custom dataset that can read from .csv or .parquet files. Includes functionality to pass in a scaler and
     transforms to preprocess dataset. Stores scaling parameters in order to apply them to test/validation datasets

     If the sequence length is set to anything but the default 0, then this functions as a time-series dataset, where
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

        self.df = read_file(file_path)
        if run_selection:
            self.df = run_select(self.df, run_selection)
        if scaler:
            self.scale_function = scaler
            self.df, self.scale_params = scaler(self.df)
        if transforms:
            self.transforms = transforms
            for transform in transforms:
                self.df = transform(self.df)

        self.sequence_length = sequence_length
        self.inputs = self.df[[col for col in self.df.columns if col.startswith(tuple(input_columns))]].fillna(0)
        self.outputs = self.df[output_columns].fillna(0)

    def __len__(self) -> int:
        return len(self.df) - self.sequence_length

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
