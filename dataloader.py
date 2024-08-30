
from typing import Any, Callable, List, Optional, Tuple

from torch.utils.data import Dataset
import pandas as pd



class CTGan_data_set(Dataset):
    def __init__(
        self, 
        data: pd.Dataframe = pd.Dataframe(),
        batch_size: int = 128,
        cond_cols: list = [],
    ):
        self.data = data
        self.cond_cols = cond_cols

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.data[idx]
        return X
    
    def columns(self) -> list:
        return list(self.data.columns)
    
    def dataframe(self) -> pd.DataFrame:
        return self.data
    
    def output_space(self) -> int:
        return len(self.data.columns)
    
    def n_units_latent(self) -> int:
        return len(self.data.columns)
    
    def n_units_conditional(self) -> int:
        return len(self.cond_cols)
    
    def cond_cols(self) -> List[str]:
        return self.cond_cols

