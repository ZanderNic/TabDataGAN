<h3 align="center">
    Conditional GAN for Tabular Data Generation
</h4>

---

<div align="center">
    [![Tabular Data](https://img.shields.io/badge/Tabular%20Data-%E2%9C%85-green)](#)
    [![Generative AI](https://img.shields.io/badge/Generative%20AI-%E2%9A%A1-yellow)](#)
    [![License: MIT](https://img.shields.io/badge/License-MIT-blue)](https://opensource.org/licenses/MIT)
</div>


## Infos:
This CTGAN repository is a project I created for fun. Feel free to use it for whatever you want! Due to time constraints, I haven’t provided a tutorial for the different loss functions and other functions, 
but feel free to explore them on your own


## :rocket: Sample Usage

```bash
$ git clone https://github.com/ZanderNic/SATGan.git
```
TODO EDIT

### Generate data

* Load and train a CTGAN

```python
from src.Model.Gans.WCTGan import WCTGan
from src.Data.dataset import CTGan_data_set
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



