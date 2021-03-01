# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd

# %%
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
sample_submission = pd.read_csv("data/sample_submission.csv")

# %%
train_df

# %%
train_df.location.value_counts(dropna=False,normalize=True)

# %%
train_df.keyword.value_counts(dropna=False,normalize=True)

# %%

# %%
import nltk  

# %% [markdown]
# ## remove stop words

# %%
from nltk.tokenize import word_tokenize

# %%
train_df["tokens"] = train_df.text.apply(lambda x: word_tokenize(x))

# %%
from nltk.corpus import stopwords


# %%
def remove_stopwords(words):
    return [word.lower() for word in words if word.lower() not in stopwords.words('english')]


# %%
train_df["tokens_clean"] = train_df["tokens"].apply(lambda x: remove_stopwords(x))

# %%
train_df

# %%
from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()
