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
import numpy as np
import gensim
from gensim.models import word2vec

# %%
train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")
sample_submission = pd.read_csv("../data/sample_submission.csv")

# %%
train_df

# %%
train_df.location.value_counts(dropna=False,normalize=True)

# %%
train_df.keyword.value_counts(dropna=False,normalize=True)

# %%
train_df.target.value_counts(dropna=False,normalize=True)

# %%
import nltk  

# %% [markdown]
# ## remove stop words

# %%
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer 
import re
lem = WordNetLemmatizer()


# %%
def preprocess_sentence(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    regex = re.compile('([^\s\w]|_)+')
    cleantext = regex.sub('', cleantext).lower()
    tokens = word_tokenize(cleantext)
    tokens_normalized = [word.lower() for word in tokens if len(word)>2 and word.lower() not in stopwords.words('english')]
    tokens_lemmatized = [lem.lemmatize(word) for word in tokens_normalized]
    return tokens_lemmatized


# %%
train_df["tokens"] = train_df["text"].map(lambda x: preprocess_sentence(x))

# %%
cv

# %%
terms_frequency = pd.DataFrame({"word": [item for sublist in np.array(train_df.tokens).tolist() for item in sublist ]})

# %%
corpus = terms_frequency.drop_duplicates().word.tolist()

# %%
model = word2vec.Word2Vec(train_df.tokens, size=100, window=20, min_count=2, workers=4)

# %%
from sklearn import feature_extraction

# %%
vectorizer = feature_extraction.text.CountVectorizer(max_features=10000, ngram_range=(1,2))

# %%
