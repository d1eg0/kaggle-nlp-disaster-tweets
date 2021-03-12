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
import unicodedata

# %%
pd.set_option('display.max_colwidth', None)

# %%
# !kaggle competitions download -c nlp-getting-started
# !unzip nlp-getting-started.zip -d ../data/

# %%
train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")
sample_submission = pd.read_csv("../data/sample_submission.csv")

# %% [markdown]
# ## First look at the datasets

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
def decontracted(phrase:str) -> str:
    # specific
    phrase = re.sub(r"wont", "will not", phrase)
    phrase = re.sub(r"cant", "can not", phrase)
    phrase = re.sub(r"wasnt", "was not", phrase)
    phrase = re.sub(r"werent", "were not", phrase)
    phrase = re.sub(r"ppl", "people", phrase)
    phrase = re.sub(r"wht", "what", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def remove_accented_chars(text: str) -> str:
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def remove_noise(sentence:str) -> str:
    sentence = re.sub(r"@", " ", sentence)
    sentence = re.sub(r"https?:\/\/\S+", " ", sentence)
    sentence = re.sub(r"[0-9]+", " ", sentence)
    return sentence

noise_words_list = ["utc","amp"]
    
def preprocess_sentence(sentence: str) -> list:
    sentence = sentence.lower()
    sentence = re.sub(r"#", ' ', sentence)
    sentence = remove_noise(sentence)
    sentence = decontracted(sentence)
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    regex = re.compile('([^\s\w]|_)+')
    cleantext = regex.sub(' ', cleantext)
    cleantext = remove_accented_chars(cleantext)
    tokens = tokenizer.tokenize(cleantext)
    tokens_normalized = [word for word in tokens if len(word)>2 and word not in stopwords.words('english')]
    tokens_lemmatized = [lem.lemmatize(word) for word in tokens_normalized if word not in noise_words_list]
    return tokens_lemmatized


# %%
def preprocess_df(df: pd.DataFrame ) -> pd.DataFrame:
    df.loc[df.keyword.isna(), "keyword"] = ""
    df.loc[df.location.isna(), "location"] = ""
    df["keyword"] = df["keyword"].astype(str)
    df["location"] = df["location"].astype(str) 
    #df["text"] = df["text"] + " " + df["keyword"].astype("str")
    df["tokens"] = df["text"].map(lambda x: preprocess_sentence(x))
    df["features"] = df.tokens.apply(lambda x: " ".join(x))
    return df


# %%
train_df = preprocess_df(train_df)

# %%
train_df

# %%
terms_frequency = pd.DataFrame({"word": [item for sublist in np.array(train_df.tokens).tolist() for item in sublist ]})

# %%
train_df[train_df.features.str.contains("amp")]

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
vectorizer = vectorizer.fit(corpus)
X = vectorizer.transform(train_df.features.to_list()) 

# %%
y = train_df["target"].to_list()

# %%
train, test = train_test_split(train_df, test_size=0.2, random_state=0)

# %%
y_train = train["target"].to_list()
y_test = test["target"].to_list()

# %%
X_train = X[train.index]
X_test = X[test.index]

# %% [markdown]
# ### dimension reduction

# %%
from sklearn import feature_selection, pipeline, naive_bayes

# %%
corpus = train.features.to_list()

# %%
X_names = vectorizer.get_feature_names()
reduce_dimension = False
if reduce_dimension:
    p_value_limit = 0.95
    dtf_features = pd.DataFrame()
    for cat in np.unique(y):
        chi2, p = feature_selection.chi2(X_train, y_train==cat)
        dtf_features = dtf_features.append(pd.DataFrame(
                       {"feature":X_names, "score":1-p, "y":cat}))
        dtf_features = dtf_features.sort_values(["y","score"], 
                        ascending=[True,False])
        dtf_features = dtf_features[dtf_features["score"]>p_value_limit]
    
    X_names = dtf_features["feature"].unique().tolist()
    vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=X_names)
    vectorizer.fit(corpus)

# %% tags=[]
X_train = vectorizer.transform(corpus)

# %% tags=[]
classifier = RandomForestClassifier(n_estimators=2000, random_state=0)
#classifier = naive_bayes.MultinomialNB()

# %%
model = pipeline.Pipeline([("vectorizer", vectorizer),  
                           ("classifier", classifier)])
model["classifier"].fit(X_train, y_train)

# %%
X_test = test["features"].values
predicted = model.predict(X_test)
y_pred = model.predict(X_test)

# %%
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

# %%
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

# %%
plot_confusion_matrix(model, X_test, y_test,
                                 display_labels=["0", "1"],
                                 cmap=plt.cm.Blues)

# %%
test.loc[:,"prediction"] = y_pred

# %%
test[test["target"] != test["prediction"]]

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
# #!kaggle competitions submit -f submission.csv -m "BoW and TIDF improved token cleaner" nlp-getting-started

# %%
# #!kaggle competitions submissions nlp-getting-started

# %%
# !pwd

# %%
