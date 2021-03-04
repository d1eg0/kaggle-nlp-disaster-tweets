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

# %% [markdown]
# ## Preprocessing

# %%
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer 
import re
lem = WordNetLemmatizer()
tokenizer = TweetTokenizer()


# %%
def preprocess_sentence(sentence: str) -> list:
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    regex = re.compile('([^\s\w]|_)+')
    cleantext = regex.sub('', cleantext).lower()
    tokens = tokenizer.tokenize(cleantext)
    tokens_normalized = [word.lower() for word in tokens if len(word)>2 and word.lower() not in stopwords.words('english')]
    tokens_lemmatized = [lem.lemmatize(word) for word in tokens_normalized]
    return tokens_lemmatized


# %%
def preprocess_df(df: pd.DataFrame ) -> pd.DataFrame:
    df["tokens"] = df["text"].map(lambda x: preprocess_sentence(x))
    df["features"] = df.tokens.apply(lambda x: " ".join(x))
    return df


# %%
train_df = preprocess_df(train_df)

# %%
terms_frequency = pd.DataFrame({"word": [item for sublist in np.array(train_df.tokens).tolist() for item in sublist ]})

# %%
terms_frequency.value_counts().head(50)

# %% [markdown]
# # Text to features

# %% [markdown]
# ## Bag of Words

# %%
from sklearn import feature_extraction
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# %%
#vectorizer = feature_extraction.text.CountVectorizer(max_features=10000, ngram_range=(1,3))
vectorizer = feature_extraction.text.TfidfVectorizer()

# %%
corpus = train_df.features.to_list()
vectorizer_fit = vectorizer.fit(corpus)
X = vectorizer_fit.transform(train_df.features.to_list()) 

# %%
y = train_df["target"].to_list()

# %%
X_train_df, X_test_df, y_train, y_test = train_test_split(train_df, y, test_size=0.2, random_state=0)

# %%
X_train = X[X_train_df.index]
X_test = X[X_test_df.index]

# %% tags=[]
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train) 

# %%
y_pred = classifier.predict(X_test)

# %%
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

# %%
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

# %%
plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=["0", "1"],
                                 cmap=plt.cm.Blues)

# %%
X_test_df.loc[:,"prediction"] = y_pred

# %%
X_test_df[X_test_df["target"] != X_test_df["prediction"]].head(50)

# %% [markdown]
# # Prediction

# %%
test_df = preprocess_df(test_df)

# %%
X_test = vectorizer_fit.transform(test_df.features.to_list()) 
y_test = classifier.predict(X_test)
#test_df['target'] = 

# %%
y_test

# %%
test_df["target"] = y_test

# %% [markdown]
# # Submission

# %%
sample_submission

# %%
submission = test_df[["id","target"]]

# %%
#submission.to_csv("submission.csv",index=False)

# %%
# #!kaggle competitions submit -f notebooks/submission.csv -m "BoW and TIDF" nlp-getting-started

# %%
# #!kaggle competitions submissions nlp-getting-started

# %%
