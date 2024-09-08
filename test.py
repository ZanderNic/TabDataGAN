import numpy as np
import pandas as pd


from CTGan import CTGan
from dataset import CTGan_data_set


n_samples = 1000


category_1 = np.random.choice(['A', 'B', 'C'], size=n_samples)
category_2 = np.random.choice(['X', 'Y', 'Z', 'W', 'V'], size=n_samples)
category_3 = np.random.choice(['M', 'N'], size=n_samples)


numeric_1 = np.random.uniform(0, 100, size=n_samples)
numeric_2 = np.random.uniform(-50, 50, size=n_samples)


df = pd.DataFrame({
    'category_1': category_1,
    'category_2': category_2,
    'category_3': category_3,
    'numeric_1': numeric_1,
    'numeric_2': numeric_2
})

data_set = CTGan_data_set(
    data=df,
    cond_cols=["category_1", "category_2"]
)


gan = CTGan(
    n_units_latent=5,
)

gan.fit(data_set)