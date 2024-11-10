from typing import Any, Callable, List, Optional, Tuple
from torch.utils.data import Dataset
import pandas as pd


class CTGan_data_set(Dataset):
    """
        This is a custom PyTorch dataset that is used for the CTGan. It implements the normal Dataset functions plus some extra functions that are needed by the CTGan.
    """

    def __init__(
        self, 
        data: pd.DataFrame = pd.DataFrame(),
        cond_cols: list = [],
        target_col: str = None,  
        cat_cols: list = [],  
        ordinal_columns: list = []  
    ):
        self._data = data
        self._cond_cols = cond_cols
        self._target_col = target_col
        self._ord_cols = ordinal_columns  
        
       
        if not cat_cols:
            if ordinal_columns:
                self._cat_cols = [
                    column for column in data.select_dtypes(include=['object', 'category', 'bool']).columns 
                    if column not in ordinal_columns
                ]
            else:
                self._cat_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        else:
            self._cat_cols = cat_cols

        if ordinal_columns:
            self._num_cols = [
                column for column in data.select_dtypes(include=['number']).columns 
                if column not in ordinal_columns
            ]
        else:
            self._num_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        for col in self._cond_cols:
            if col not in data.columns:
                raise ValueError(f"Conditional column '{col}' is not present in the dataset.")


    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        X = self._data.iloc[idx]
        return X
    
    def columns(self) -> list:
        return list(self._data.columns)
    
    def dataframe(self) -> pd.DataFrame:
        return self._data
    
    def df(self) -> pd.DataFrame:
        return self._data

    def cond_cols(self) -> List[str]:
        return self._cond_cols

    def cat_cols(self):
        return self._cat_cols

    def num_cols(self):
        return self._num_cols
    
    def ord_cols(self):
        return self._ord_cols
