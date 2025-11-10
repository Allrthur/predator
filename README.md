<img src=".github/logo.svg" width=185 height=185 align="right">

# PREDATOR

> **PRE-trained Data AugmenTOR**

This is a for from the code release for [Pre-trained Data Augmentation for Text Classification](https://link.springer.com/chapter/10.1007/978-3-030-61377-8_38). Adapted for use in [Here] (link)


## Usage

```python
import torch
import pandas as pd
from predator import Predator

df_train = pd.read_csv("train.csv") # => containing 'text' and 'label' columns
df_valid = pd.read_csv("valid.csv")

device = "cuda" if torch.cuda.is_available() else "cpu"

predator = Predator(df_train, df_valid, device=device)
predator.train()

df_aug = predator.augment(augment_ratio=3.0) # => augmented dataset
df_train = pd.concat([df_train,df_aug]) # concatenate generated samples with dataset
```

## Installation

You can install this package via pip.

`pip install git+https://github.com/hugoabonizio/predator`


## Citation

```
@inproceedings{abonizio2020,
  author="Hugo Queiroz Abonizio and Sylvio Barbon Junior",
  editor="Cerri, Ricardo and Prati, Ronaldo C.",
  title="Pre-trained Data Augmentation for Text Classification",
  booktitle="Intelligent Systems",
  year="2020",
  publisher="Springer International Publishing",
  address="Cham",
  pages="551--565",
  isbn="978-3-030-61377-8"
}
```