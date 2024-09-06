
from typing import Any, Callable, List, Optional, Tuple

from torch.utils.data import Dataset
import pandas as pd


class CTGan_data_set(Dataset):
    """
        This is a custom pytorch dataset that is used for the CTGan it implemnets the normal Dataset funktions plus some extra funktions that are needed by the CTGan
    """

    def __init__(
        self, 
        data: pd.Dataframe = pd.Dataframe(),
        cond_cols: list = [],
        target_col : str = None,
        ordinal_columns = None
    ):
        self.data = data
        self.cond_cols = cond_cols
        self.target_col = target_col
        self.ord_cols = ordinal_columns # If this parameter is not given one cant distinguish betwean ord and cat cols or num cols
        self.cat_cols = [column for column in data.select_dtypes(include=['object', 'category', 'bool']).columns if column not in ordinal_columns]
        self.num_cols = data.select_dtypes(include=['number']).columns.tolist()    

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.data[idx]
        return X
    
    def columns(self) -> list:
        return list(self.data.columns)
    
    def dataframe(self) -> pd.DataFrame:
        return self.data
    
    def cond_cols(self) -> List[str]:
        return self.cond_cols

    def cat_cols(self):
        return self.cat_cols

    def num_cols(self):
        return self.cat_cols
    
    def ord_cols(self):
        return self.ord_cols
