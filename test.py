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

from src.Model.Gans.WCTGan import WCTGan
from src.Data.dataset import CTGan_data_set
from src.Benchmark.benchmark import Benchmark

from src.Model.Generators.Generator import Conv_Generator
from src.Model.Critic.critic import Conv_Discriminator


iris = load_iris(as_frame=True)
df = iris['frame']

data_set = CTGan_data_set(
    data=df,
    cond_cols=["target"],
    cat_cols=["target"]  
)


wctgan = WCTGan(
    n_units_latent=500
                
                
)#generator_class=Conv_Generator, discriminator_class=Conv_Discriminator)


crit_loss, gen_loss = wctgan.fit(
    data_set, 
    n_epochs=500, 
    discriminator_num_steps=5,
    generator_num_steps=1,
    lambda_gradient_penalty=10,
)

plot_gan_losses(crit_loss, gen_loss)

## Generate new data
cond_df = pd.DataFrame([{"target" : 1}]*160)
syn_df = wctgan.gen(160)#cond_df=cond_df)

print(syn_df)

# Benchmark the generated data 
benchmark = Benchmark()
mean_rfc = benchmark.mean_rfc(df, syn_df, plot=True)

print(f"Mean RFC= {mean_rfc}")