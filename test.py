import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import sys
import os

def plot_gan_losses(crit_loss, gen_loss,
                    critic_color='blue',
                    gen_color='orange',  
                    save_path="train_loss.png"
    ):
    
    plt.figure(figsize=(10, 6))

    plt.plot(crit_loss, label="Critic Loss", color=critic_color, linestyle='-', linewidth=1.5)
    plt.plot(gen_loss, label="Generator Loss", color=gen_color, linestyle='-', linewidth=1.5)
  
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Validation Losses for Generator and Critic")
    plt.legend()

    plt.savefig(save_path)

# This Projekt imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Model.WCTGAN import CTGan
from src.Data.dataset import CTGan_data_set
from src.Benchmark.benchmark import Benchmark

iris = load_iris(as_frame=True)
df = iris['frame']


data_set = CTGan_data_set(
    data=df,
    cond_cols=["target"],
    cat_cols=["target"]  
)

gan = CTGan()

crit_loss, gen_loss = gan.fit(data_set)

plot_gan_losses(crit_loss, gen_loss)


cond_df = pd.DataFrame([{"target" : 1}]*150)

syn_df = gan.gen(150, cond_df=cond_df)

print(syn_df)

benchmark = Benchmark()

print(benchmark.mean_rfc(df, syn_df))