from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
import numpy as np
import urllib.request
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from bm25 import bm

app = Flask(__name__)


df = pd.read_json('News_Dataset.json', lines = True)
df = df.drop_duplicates() 
cols = ['headline', 'short_description']
df['combined'] = df[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)



words = set(nltk.corpus.words.words())

def preprocess_text(text):
    str1 = " "
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = " ".join(w for w in nltk.wordpunct_tokenize(text) \
         if w.lower() in words or not w.isalpha())
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered = [w for w in word_tokens if not w.lower() in stop_words]
    com = str1.join(filtered)
    return com

df["combined"] = df.combined.apply(preprocess_text)

vectorizer = CountVectorizer(stop_words='english')
documents_vectorized = vectorizer.fit_transform(df['combined'])
vocabulary = vectorizer.get_feature_names_out()
vocabulary
print("")
print("")
print ('We have a {} document corpus with a {} term vocabulary'.format(*documents_vectorized.shape))
print("")
print("")

dataframe = pd.DataFrame(documents_vectorized.toarray(), columns=vocabulary)
doc_ids = dataframe.index.values

def BM25_IDF_df(df):
  """
  This definition calculates BM25-IDF weights before hand as done last week
  """

  dfs = (df > 0).sum(axis=0)
  N = df.shape[0]
  idfs = -np.log(dfs / N)
  
  k_1 = 1.2
  b = 0.8
  dls = df.sum(axis=1) 
  avgdl = np.mean(dls)

  numerator = np.array((k_1 + 1) * df)
  denominator = np.array(k_1 *((1 - b) + b * (dls / avgdl))).reshape(N,1) \
                         + np.array(df)

  BM25_tf = numerator / denominator

  idfs = np.array(idfs)

  BM25_score = BM25_tf * idfs
  return pd.DataFrame(BM25_score, columns=vocabulary)



bm25_df = BM25_IDF_df(dataframe[:1000])

# for i in range(1000, 209514, 1000):
#     bm25_df1 = BM25_IDF_df(dataframe[:i])  # a dataframe with BM25-idf weights
#     bm25_df.append(bm25_df1, ignore_index = True)
    
for i in range(1000, 1000, 1000):
    bm25_df1 = BM25_IDF_df(dataframe[:i])  # a dataframe with BM25-idf weights
    bm25_df.append(bm25_df1, ignore_index = True)


def retrieve_ranking(query, bm25_df):
  q_terms = query.split(' ')
  q_terms_only = bm25_df[q_terms]
  score_q_d = q_terms_only.sum(axis=1)
  return sorted(zip(bm25_df.index.values, score_q_d.values),
                key = lambda tup:tup[1],
                reverse=True)



@app.route("/", methods =["GET", "POST"])
def home():
    if request.method == "POST":
       # getting input with name = fname in HTML form
       final = []
       query1 = request.form.get("search")
       query = preprocess_text(query1)
       try:
          results = retrieve_ranking(query, bm25_df)
       except:
          print("this query is not in out index")
          return render_template("errorMessage.html", query = query1)
       for i in results[:10]:
         final.append(i[0])
       print(final)
       
       headline = df.loc[final, 'headline']
       link = df.loc[final, 'link']
       disc = df.loc[final, 'short_description']
    else:
       link = []
       headline =[]
       final = []
       disc = []
    return render_template("index.html", link = link, headline= headline, final = final, disc = disc)


@app.route("/results")
def res():
    return render_template("results.html")


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
