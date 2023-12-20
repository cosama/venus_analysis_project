import torch
from torch.utils.data import Dataset
import pandas as pd


class VenusBaseDataset(Dataset):
    def __init__(self, parquet_file, input_columns, output_columns):
        """
        Base Class that implements feature/target selection
        and __len__ / __getitem__ methods
        Args:
            parquet_file (string): Path to the parquet file.
            input_columns (list of string): List of column names to be used as input features.
            output_columns (list of string): List of column names to be used as output targets.
        """
        self.df = pd.read_parquet(parquet_file)
        self.input_columns = input_columns
        self.output_columns = output_columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = self.df.iloc[idx][self.input_columns]
        outputs = self.f.iloc[idx][self.output_columns]

        inputs = torch.tensor(inputs.values, dtype=torch.float32)
        outputs = torch.tensor(outputs.values, dtype=torch.float32)

        return inputs, outputs
