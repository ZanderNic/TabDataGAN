import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# This Projekt imports
from WCTGAN import CTGan
from dataset import CTGan_data_set


iris = load_iris(as_frame=True)
df = iris['frame']

print(df)

data_set = CTGan_data_set(
    data=df,
    cond_cols=["target"],
    cat_cols=["target"]  
)

gan = CTGan()
loss_crit, loss_gan  = gan.fit(data_set)

plt.plot(loss_crit, label="Critic Loss")
plt.plot(loss_gan, label="Generator Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("train.png")

cond_df = pd.DataFrame([{"target" : 1}]*100)

print(gan.gen(100, cond_df=cond_df))
