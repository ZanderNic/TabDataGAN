import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


# This Projekt imports
from WCTGAN import CTGan
from dataset import CTGan_data_set




iris = load_iris(as_frame=True)
df = iris['frame']


print(df)

data_set = CTGan_data_set(
    data=df,
    cond_cols=["target"],  
)

gan = CTGan(n_units_latent=5)
gan.fit(data_set)

cond_df = pd.DataFrame([1]*100)


print(gan.gen(100, cond_df=cond_df))
