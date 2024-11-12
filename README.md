---

<h3 align="center">
    Conditional GAN for Tabular Data Generation
</h4>

---

<p align="center">
    <a href="#"><img src="https://img.shields.io/badge/Tabular%20Data%20Generation-%E2%9C%85-green"></a>
    <a href="#"><img src="https://img.shields.io/badge/Conditional%20Generation-%F0%9F%94%A5-orange"></a>
    <a href="#"><img src="https://img.shields.io/badge/Generative%20AI-%E2%9A%A1-yellow"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue"></a>
</p>



## Infos
This repository hosts a Tabular GAN project that I created for fun and learning. It was inspired by the CTGAN paper ([Paper](https://arxiv.org/pdf/1907.00503)) and other works in the GAN context, including CTAB-GAN 
([Paper](https://arxiv.org/abs/2102.08369)). My implementation supports conditional data generation and includes several custom loss functions and additional features that I experimented with.


It's worth noting that GANs can be quite challenging to train effectively, especially for tabular data. This difficulty is illustrated in Figure 1 of the GReaT paper ([Paper](https://openreview.net/forum?id=cEygmQNOeI)), which demonstrates how even simple datasets can be difficult to model with GANs.

Due to time constraints, I haven’t included a detailed tutorial on the various loss functions and features, but feel free to explore the code and try them out on your own.


## :rocket: Sample Usage

```bash
$ git clone https://github.com/ZanderNic/SATGan.git
$ pip install .
```


### Generate data

* Load and train a CTGAN


```python
from table_gan.Model.Gans.WCTGan import WCTGan
from table_gan.Data.dataset import CTGan_data_set
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris['frame']

data_set = CTGan_data_set(
    data=df,
    cond_cols=["target"],
    cat_cols=["target"]  
)

wctgan = WCTGan()

crit_loss, gen_loss = wctgan.fit(
    data_set, 
    n_epochs=10, 
)
```

* Generate new synthetic tabular data

```python

cond_df = pd.DataFrame([{"target" : 1}]*160)
syn_df = wctgan.gen(cond_df=cond_df)
```


## Credits and Acknowledgments

This project is inspired by [CTGAN](https://arxiv.org/pdf/1907.00503) and the methods described in their work.

